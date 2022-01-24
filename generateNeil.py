# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
from torch import lerp

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

import random, copy
import math
from tqdm import tqdm

#for slerp
from numpy import asarray
from numpy import arccos
from numpy import clip
from numpy import dot
from numpy import sin
from numpy import linspace
from numpy.linalg import norm

from enum import Enum
class Animator(Enum):
    SIMPLE = 1
    GENERATIONS = 2
    POINT_EXPLORER = 3

class DNA:
    def __init__(self, G, generations, seed=None,generation_seeds=None):
        self.genes = create_random_vector(G,seed)
        self.og_genes = create_random_vector(G,seed)
        self.generations = []
        self.num_generations = generations-1
        for i in range(generations):
            if generation_seeds:
                this_seed = generation_seeds[i]
                print(f"Generation {i} \t seed {this_seed}")
                self.generations.append(create_random_vector(G, seed=this_seed))
            else:
                self.generations.append(create_random_vector(G))

    def crossover(self, partner,mutation_rate):
        child = copy.deepcopy(self)
        for i in range(len(self.genes[0])):
            child.genes[0][i] = linear_interpolate(child.genes[0][i],partner.genes[0][i],mutation_rate)
        return child

    def mutate(self, mutation_rate):
        mutation_speed = 0.1
        for i in range(len(self.genes)):
            if i % 2 == 0:
                self.genes[0][i] += mutation_speed
            else:
                self.genes[0][i] -= mutation_speed
            #self.genes[0][i] = np.random.RandomState().randn()

    def mutate_target(self,target,speed):
        for i in range(len(self.genes)):
            if i % 2 == 0:
                self.generations[target][0][i] += speed*15
            else:
                self.generations[target][0][i] -= speed*15

    def mutate_generation(self,overall_progress, evolve=None,frames=None):
        #overall = 0-1 for sequence
        # 1 / 4 = 0.25
        sequence_time = 1 / len(self.generations)
        steps = math.floor(frames * sequence_time)

        generation_target = overall_progress / sequence_time
        generation_target = math.floor(generation_target)

        #0.24 / 4 = 0.06
        step_size = sequence_time / len(self.generations)
        step_size *= 2
        #1-0.06 = 0.94
        lerp = np.clip(1-step_size, 0, 1)
        #lerp = pow(lerp, 0.5)


        if evolve:
            self.mutate_target(generation_target, step_size)
        for i in range(len(self.genes[0])):
            linear = True
            if linear:
                self.genes[0][i] = linear_interpolate(self.genes[0][i],
                                                      self.generations[generation_target][0][i],
                                                      lerp)
            else:
                vectors = interpolate_points(self.genes[0][i],
                                             self.generations[generation_target][0][i],
                                             1-lerp)
                self.genes[0][i] = vectors


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    from numpy.random import randn
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input


def linear_interpolate(alpha,code1, code2):
    return (alpha * code1) + ((1 - alpha) * code2)
    #return code1 * alpha + code2 * (1 - alpha)


def slerp(val, low, high):
    omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
    so = sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val) * low + val * high
    return sin((1.0-val)*omega) / so * low + sin(val*omega) / so * high

def slerp2(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return linear_interpolate(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    if c:
        res = torch.from_numpy(v2).to("cuda")
    else:
        res = v2
    return res

def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = slerp2(ratio, p1, p2)
        vectors.append(v)
    return vectors


def generate_initial_population(quant, G,generations, seed=None,gen_seeds=None):
    if seed:
        return [DNA(G, generations, seed=(seed + _), generation_seeds=gen_seeds) for _ in range(quant)]
    else:
        return [DNA(G, generations, generation_seeds=gen_seeds) for _ in range(quant)]

def _build_mating_pool(population, fitness):
    mating_pool = []
    for i in range(len(population)):
        mating_pool += [population[i]] * fitness[i]
    return mating_pool

def evolve(population, fitness, mutation_rate):
    new_population = []
    for _ in range(len(population)):
        mother = random.choice(population)
        father = random.choice(population)
        child = mother.crossover(father, mutation_rate)
        child.mutate(mutation_rate)
        new_population.append(child)
    return new_population

def evolve_generation(population,sequence_length,cur_frame,evolve=None):
    over_all_progress = cur_frame/sequence_length
    for child in population:
        child.mutate_generation(over_all_progress,evolve=evolve,frames=sequence_length)
    return population

def create_random_vector(G,seed=None):
    if seed:
        return np.random.RandomState(seed).randn(1, G.z_dim)
    else:
        return np.random.RandomState().randn(1, G.z_dim)


def generate_image(G, latent_vector,psi=None):
    device = torch.device("cuda")
    # os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)
    if not psi:
        psi = 1
    z = torch.from_numpy(np.array(latent_vector)).to(device)
    img = G(z, label, truncation_psi=psi, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img
    # filename = f'{outdir}/{str(time.time())}.png'
    #return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")  # .save(filename)


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--gen', type=num_range, help='How many images to cycle through')
@click.option('--frames', type=num_range, help='sequence length')
@click.option('--steps', type=num_range, help='steps between generations')
@click.option('--gen_seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    gen_seeds: Optional[List[int]],
    gen : int,
    frames : int,
    steps : int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    #original generate images
    # Generate images.
    use_original = False

    sequence_length = 250
    if frames:
        sequence_length = frames[0]


    animator = Animator.POINT_EXPLORER


    if animator == Animator.GENERATIONS:
        for seed_idx, seed in enumerate(seeds):
            this_outdir = f"{outdir}/seed_{seed}"
            os.makedirs(this_outdir, exist_ok=True)
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            population_size = 1
            if gen_seeds:
                generations = len(gen_seeds)
            elif gen:
                generations = gen[0]
            else:
                generations = 5
            population = generate_initial_population(population_size, G, generations, seed=seed,gen_seeds=gen_seeds)
            print(f"Creating sequence with length {sequence_length}\n"
                  f"Number of generations {generations}\n")
            for i in range(population_size):
                starting_psi = 0.3
                psi_variation = 0.7
                for run in tqdm(range(sequence_length)):
                    #mutation_rate += 1/sequence_length
                    starting_psi += psi_variation / sequence_length
                    out_psi = 1 - starting_psi
                    if truncation_psi:
                        out_psi = truncation_psi
                    img = generate_image(G, population[i].genes,psi=out_psi)
                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{this_outdir}/seed{seed:04d}_pop{i:02d}_run{run:04d}.png')
                    population = evolve_generation(population, sequence_length, run,evolve=True)
    elif animator == Animator.SIMPLE:
        for seed_idx, seed in enumerate(seeds):
            this_outdir = f"{outdir}/seed_{seed}"
            os.makedirs(this_outdir, exist_ok=True)
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            rand = np.random.RandomState(seed).randn(1, G.z_dim)
            z = torch.from_numpy(rand).to(device)
            starting_psi = 0.1
            seed_blend = 5000
            seed_progress = 0
            num_runs = sequence_length
            psi_variation = 0.75
            for run in tqdm(range(num_runs)):
                seed_progress += seed_blend / num_runs
                #z = torch.from_numpy(rand + seed_progress).to(device)
                rand[0][20] += seed_progress
                rand[0][30] += seed_progress
                rand[0][40] -= seed_progress
                z = torch.from_numpy(rand).to(device)

                starting_psi += psi_variation / num_runs
                out_psi = 1-starting_psi
                if truncation_psi:
                    out_psi = truncation_psi
                img = G(z, label, truncation_psi=out_psi, noise_mode=noise_mode,)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{this_outdir}/seed{seed:04d}_run{run:04d}.png')
    elif animator == Animator.POINT_EXPLORER:
        for seed_idx, seed in enumerate(seeds):
            targets = list()
            draw_vectors = list()
            #create the first picture.
            targets.append(create_random_vector(G,seed))
            if gen_seeds:
                generations = len(gen_seeds)
            elif gen:
                generations = gen[0]
            else:
                generations = 5
            #create a range of new points to go to.
            for i in range(generations+1):
                if gen_seeds:
                    targets.append(create_random_vector(G,gen_seeds[i]))
                else:
                    targets.append(create_random_vector(G))

            #animation steps
            if steps:
                steps = steps[0]
            elif sequence_length:
                steps = int(sequence_length/generations)
            else:
                steps = 25

            for target_idx, target in enumerate(targets):
                this_target = target
                next_target = target
                try:
                    next_target = targets[target_idx+1]
                except:
                    pass
                intermediate = interpolate_points(target,next_target,steps)
                draw_vectors = draw_vectors + intermediate
#                draw_vectors.extend(intermediate)
                pass

            this_outdir = f"{outdir}/seed_{seed}"
            os.makedirs(this_outdir, exist_ok=True)
            for vector_id, vector in tqdm(enumerate(draw_vectors)):
                z = torch.from_numpy(vector).to(device)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, )
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{this_outdir}/seed{seed:04d}_run{vector_id:04d}.png')

        pass


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
