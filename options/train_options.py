import argparse

class TrainOptions:
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Train options for TraceTrans model")
        
        parser.add_argument('--exp-name', required=True, help='experiment name')
        parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')

        # data organization parameters
        parser.add_argument('--pairlist-path', required=True, help='pair list for retrieving training data')

        # training parameters
        parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
        parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
        parser.add_argument('--epochs', type=int, default=100,
                            help='number of training epochs (default: 100)')
        parser.add_argument('--steps-per-epoch', type=int, default=45,
                            help='number of training steps per epoch (default: 45)')
        parser.add_argument('--ckpt-path', type=str, help='path to saved checkpoint')
        parser.add_argument('--initial-epoch', type=int, default=0,
                            help='initial epoch number (default: 0)')
        parser.add_argument('--lr-G', type=float, default=2e-4, help='learning rate for generator (default: 2e-4)')
        parser.add_argument('--lr-D', type=float, default=2e-4, help='learning rate for discriminator (default: 2e-4)')
        parser.add_argument('--beta-G', type=float, default=0.9, help='beta in Adam optimizer for generator')
        parser.add_argument('--beta-D', type=float, default=0.9, help='beta in Adam optimizer for discriminator')
        parser.add_argument('--cudnn-nondet', action='store_true', help='disable cudnn determinism - might slow down training')
        
        # network architecture parameters
        parser.add_argument('--input-nc', type=int, default=1, help='number of channels of input image, default is 1 for grayscale image')
        parser.add_argument('--output-nc', type=int, default=2, help='number of channels for flow field')
        parser.add_argument('--integration-steps', type=int, default=7, 
                            help='number of integration steps applied to flow field')
        parser.add_argument('--ndf', type=int, default=64, 
                            help='number of downsampling filters, representing the number of filters in the first conv layer')
        parser.add_argument('--net-D', default='basic', help='architecture for discriminator, default \'basic\'')

        # loss parameters
        parser.add_argument('--gan-mode', default='lsgan', help='GAN loss name')
        parser.add_argument('--alpha', type=float, default=1, help='weight for norm of gradient of flow field')
        parser.add_argument('--beta', type=float, default=0.005, help='weight for discriminator loss')
        
        parser.add_argument('--img-size', type=int, default=256, help='size of the input image')
        parser.add_argument('--use-rgb', action='store_true', help='use 3-channel colored images')
        parser.add_argument('--gen-proportion', type=float, default=0.5, help='the proportion of generation stream (default: 0.5)')
        
        # use_depth
        parser.add_argument('--use-depth', type=bool, default=False, help='whether to use depth map as condition')

        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()