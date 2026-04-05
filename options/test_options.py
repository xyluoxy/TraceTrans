import argparse

class TestOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Validate options for TraceTrans model")
        
        # key parameters
        parser.add_argument('--exp-name', required=True, help='experiment name')
        parser.add_argument('--ckpt-path', required = True, type=str, help='path to saved checkpoint')
        parser.add_argument('--pairlist-path', required=True, help='pair list for retrieving testing data')
        parser.add_argument('--output-path', required=True, help='output path')
        
        # registering parameters
        parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
        parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
        parser.add_argument('--cudnn-nondet', action='store_true',
                            help='disable cudnn determinism - might slow down training')
        
        # network architecture parameters
        parser.add_argument('--input-nc', type=int, default=1, help='number of channels of input image, default is 1 for grayscale image')
        parser.add_argument('--output-nc', type=int, default=2, help='number of channels for flow field')
        parser.add_argument('--integration-steps', type=int, default=7, help='number of integration steps applied to flow field')
        
        
        parser.add_argument('--use-rgb', action='store_true', help='use 3-channel colored images')
        
        # use_depth
        parser.add_argument('--use-depth', type=bool, default=False, help='whether to use depth map as condition')

        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()