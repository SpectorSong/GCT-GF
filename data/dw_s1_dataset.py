"""Dataset class SEN12MSCRTS

This class wraps around the SEN12MSCRTS dataloader in ./dataLoader.py
"""

from data.base_dataset import BaseDataset
from data.dataLoader_nearest import DW_S1


class DWS1Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.data_loader = DW_S1(opt.dataroot, split=opt.input_type, sample_type=opt.sample_type,
                                 n_input_samples=opt.n_input_samples)
        self.max_bands = 3

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        nearest = self.data_loader.__getitem__(index)
        A_DW, A_DW_mask = [], []
        #todo： 实验时补充
        # if self.opt.in_only_S1:  # using only S1 input
        #     for i in range(self.opt.n_input_samples):
        #         A_DW_01 = cloudy_cloudfree['input']['S1'][i]
        #         if self.rescale_method == 'default':
        #             A_DW.append((A_DW_01 * 2) - 1)  # rescale from [0,1] to [-1,+1]
        #         elif self.rescale_method == 'resnet':
        #             A_DW.append(A_DW_01)  # no need to rescale, keep at [0,5]
        #         A_DW_mask.append(cloudy_cloudfree['target']['masks'][0].reshape((1, 256, 256)))
        # else:  # this is the typical case
        for i in range(self.opt.n_input_samples):
            A_DW_01 = nearest['input']['DW'][i]
            A_DW.append(A_DW_01)
            A_DW_mask.append(nearest['input']['masks'][i].reshape((1, 256, 256)))
        B = nearest['target']['DW'][0]

        if self.opt.s1_channels == 'vv':
            c_index = 0
        elif self.opt.s1_channels == 'vh':
            c_index = 1
        else:
            c_index = 2
        A_S1 = []
        for i in range(self.opt.n_input_samples):
            if c_index > 1:
                A_S1_01 = nearest['input']['S1'][i]
            else:
                A_S1_01 = nearest['input']['S1'][i][c_index]
            A_S1.append(A_S1_01)  # no need to rescale, keep at [0,2]
        if c_index > 1:
            B_S1 = nearest['target']['S1'][0]
        else:
            B_S1 = nearest['target']['S1'][0][c_index]

        return {'A_S1': A_S1, 'A_DW': A_DW, 'A_mask': A_DW_mask, 'B': B, 'B_S1': B_S1, 'patch_id': nearest['patch_id']}

    def __len__(self):
        """Return the total number of images."""
        return len(self.data_loader)

