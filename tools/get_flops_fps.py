'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/tools/get_flops.py
'''
import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor

import torch
import warnings
warnings.filterwarnings('ignore')

def fps_params_flops(model, size):
    import time
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    iterations = None
    input = torch.randn(size).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS, ">>>res. ", size)
    
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis,flop_count_str,flop_count_table

    flops = FlopCountAnalysis(model, input)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, input)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    # print(flop_count_table(flops))
    # print(f"module flops : {flops.by_module_and_operator()}")
    print(f"number of parameter: {param}")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int, 
        nargs='+',
        default=[1024, 2048],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape,ost=open('flops_freer_full.txt','w'))
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    
    

    sizes = [[1,3,1024, 2048]]
    for size in sizes:
        fps_params_flops(model, size)

if __name__ == '__main__':
    main()
