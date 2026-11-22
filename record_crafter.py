"""Record one real Crafter episode from a trained Dreamer checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tensordict import TensorDict

import tools
from dreamer import Dreamer
from envs import make_env


def observation_batch(obs, reward, device):
    """Match the single-environment observation layout used during training."""
    values = {key: torch.as_tensor(value, device=device).unsqueeze(0) for key, value in obs.items()}
    values['reward'] = torch.tensor([reward], dtype=torch.float32, device=device)
    for key, value in values.items():
        if value.ndim == 1:
            values[key] = value.unsqueeze(-1)
    return TensorDict(values, batch_size=(1,), device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()
    config = OmegaConf.load(args.run / '.hydra/config.yaml')
    config.model.compile = False
    config.seed = args.seed
    tools.set_seed_everywhere(args.seed)
    env = make_env(config.env, 0)
    try:
        agent = Dreamer(config.model, env.observation_space, env.action_space).to(config.device)
        checkpoint = torch.load(args.run / 'latest.pt', map_location='cpu', mmap=True, weights_only=False)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.eval()
        obs = env.reset()
        state = agent.get_initial_state(1)
        reward, total, step, done = 0.0, 0.0, 0, False
        action_name = 'reset'
        frames = []
        actions = []
        import crafter

        while True:
            canvas = Image.new('RGB', (512, 576), '#161b22')
            image = Image.fromarray(obs['image']).resize((512, 512), Image.Resampling.NEAREST)
            canvas.paste(image, (0, 64))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), 'DreamerV3 size12M | 100k training steps | REAL gameplay', fill='white')
            draw.text((12, 29), f'Step: {step}  Return: {total:.1f}  Action: {action_name}', fill='white')
            draw.text((12, 46), f'Playback: {args.fps} FPS | Seed: {args.seed}' + (' | EPISODE ENDED' if done else ''), fill='white')
            frames.append(np.asarray(canvas))
            if done:
                break
            with torch.inference_mode():
                action, state = agent.act(observation_batch(obs, reward, config.device), state, eval=True)
            action_np = action[0].cpu().numpy()
            action_name = crafter.constants.actions[int(action_np.argmax())]
            actions.append(action_name)
            obs, reward, done, info = env.step(action_np)
            total += float(reward)
            step += 1

        output = args.run / f'gameplay_seed{args.seed}.mp4'
        clip = ImageSequenceClip(frames, fps=args.fps)
        try:
            clip.write_videofile(str(output), codec='libx264', audio=False, fps=args.fps, logger=None,
                                 ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        finally:
            clip.close()
        summary = {
            'checkpoint': str(args.run / 'latest.pt'), 'seed': args.seed,
            'steps': step, 'return': total, 'fps': args.fps,
            'duration_seconds': len(frames) / args.fps,
            'final_health': int(info['inventory']['health']),
            'achievements': {k: int(v) for k, v in info['achievements'].items() if v > 0},
            'actions': actions,
        }
        output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
        Image.fromarray(frames[len(frames) // 2]).save(output.with_suffix('.png'))
        print(json.dumps({k: v for k, v in summary.items() if k != 'actions'}, indent=2))
        print('Video:', output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
