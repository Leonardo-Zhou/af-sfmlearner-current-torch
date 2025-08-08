from __future__ import absolute_import, division, print_function

from trainer_stage_two import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


def main():
    """主函数，支持命令行参数和直接参数设置"""
    import sys
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 使用命令行参数
        options = MonodepthOptions()
        opts = options.parse()
    else:
        # 调试模式：直接设置参数
        options = MonodepthOptions()
        
        # 设置调试用的参数（根据你的需求修改）
        debug_args = [
            '--data_path', '/mnt/data/publicData/MICCAI19_SCARED/train',
            '--log_dir', 'dpao_model',
            '--load_weights_folder', './optical_flow_model'
            # 你可以在这里添加更多参数，例如：
            # '--batch_size', '8',
            # '--learning_rate', '1e-4',
            # '--num_epochs', '20'
        ]
        
        opts = options.parse(debug_args)
    
    trainer = Trainer(opts)
    trainer.train()


if __name__ == "__main__":
    main()