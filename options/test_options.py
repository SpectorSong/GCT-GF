from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='G:\\reconstruction\\result', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavior during training and test.
        parser.add_argument('--eval', type=bool, default=True, help='use eval mode during test time.')
        parser.add_argument('--various_gap_coverage', action='store_true', help='test in different coverage range')
        parser.add_argument('--test_id', type=int, default=0, help='patch id, used only in analysis')
        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        # self.print_options(parser.parse_args(), 'test')
        self.isTrain = False
        return parser
