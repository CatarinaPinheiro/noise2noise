import h5py
import numpy as np


class DictsArray:
    def __init__(self, hdf5filepath, strides, shotsize_list,\
                data_keys, cropx, input_extension='_noisy',\
                output_extension='_clean', train_data=0.4, val_data=0.1):
        """
        Generate crops to using in input data
        Args:
            hdf5filepath (string): hdf5 path
            cropx (int): crop size in x
            data_keys (list): List of datakeys using as dataset
            shotsize_list (list): shot size for the seismic
            strides (list): stride size
            input_extension(str): extension of input path to find the file
            output_extension(str): extension of output path to find the file
            train_data (float): % of data used in training
            val_data (float): % of data used in validation
        """

        np.random.seed(0) # generate a random seed

        self.hdf5filepath = hdf5filepath
        self.strides = strides
        self.data_keys = data_keys
        self.shotsize_list = shotsize_list
        self.input_extension = input_extension
        self.output_extension = output_extension
        self.train_data = train_data
        self.val_data = val_data

        self.cropsize = [cropx, min(self.shotsize_list)]

        # Mean and Std calculation
        with h5py.File(self.hdf5filepath, 'r') as hdf5:
            self.inputs_shape = []
            sum_datasets = []
            len_datasets = []

            for data_key in self.data_keys:
                dataset = hdf5[data_key + self.input_extension]
                self.inputs_shape.append(dataset.shape)
                used_data = dataset[:,:int(dataset.shape[1]*self.train_data)].flatten()
                sum_data = np.sum(used_data)
                len_data = len(used_data)
                sum_datasets.append(sum_data)
                len_datasets.append(len_data)
            print("Inputs shape", self.inputs_shape)
            self.trainMean = np.sum(sum_datasets)/np.sum(len_datasets)
            sum_data_std = []
            for data_key in self.data_keys:
                dataset = hdf5[data_key + self.input_extension]
                used_data = dataset[:,:int(dataset.shape[1]*self.train_data)].flatten()
                sum_data_std.append(np.sum(abs(used_data - self.trainMean)**2))

            self.trainStd = np.sqrt(np.sum(sum_data_std)/ np.sum(len_datasets))

    def get_xy (self, data_position, n_initial, n_used_data):
        """
        Get x and y values to build crop ofrom each input dataset. 
        It's important that crosize doesn't exceed the shotsize value.
        """
        x_length = self.inputs_shape[data_position][0]
        last_x_start = x_length - self.cropsize[0] 
        y_length = self.inputs_shape[data_position][1]*n_used_data
        y_initial = int(self.inputs_shape[data_position][1]*n_initial/ self.shotsize_list[data_position])*\
                        self.shotsize_list[data_position]
        n_shots_dataset = int(y_length/ self.shotsize_list[data_position])

        # Calculate xstart and ystart without repetition
        xs = np.arange(0, last_x_start+1, self.strides[0])
        if x_length % self.cropsize[0] != 0:
            xs = np.append(xs, (x_length - self.cropsize[0]))
        ncrops_x = len(xs)

        ys = np.arange(y_initial, (y_initial + self.shotsize_list[data_position]), self.strides[1])
        ny = np.arange(n_shots_dataset)
        y_start_list = list(map(lambda a: ys + self.shotsize_list[data_position]*a, ny))
        y_start_conc = np.concatenate(y_start_list)
        ncrops_y = len(y_start_conc)

        total_crops = ncrops_x*ncrops_y
        print('Number of crops: ', total_crops)

        # Calculate xstart and ystart with repetition and calculate xend and yend
        x_start = np.repeat(xs, ncrops_y)
        y_start = np.tile(y_start_conc, ncrops_x)
        x_end = x_start + self.cropsize[0]
        y_end = y_start + self.cropsize[1]

        # Define a dataset type
        dataset_type = np.repeat(self.data_keys[data_position], total_crops)

        zip_crops = list(zip(dataset_type, x_start, x_end, y_start, y_end))

        return zip_crops

    def dicts_train(self):
        data_dicts = []
        for i in np.arange(len(self.data_keys)):
            data_dict = self.get_xy(i, n_initial=0, n_used_data=self.train_data)
            data_dicts.append(data_dict)
        train_dicts = [x for xs in data_dicts for x in xs]
        train_dicts_array = [{'dataset': data, 'xstart': a, 'xend': b,\
                              'ystart': c, 'yend': d} for data, a, b, c, d in train_dicts]
        
        return train_dicts_array

    def dicts_val(self):
        data_dicts = []
        for i in np.arange(len(self.data_keys)):
            data_dict = self.get_xy(i, n_initial=self.train_data, n_used_data=self.val_data)
            data_dicts.append(data_dict)
        val_dicts = [x for xs in data_dicts for x in xs]
        val_dicts_array = [{'dataset': data, 'xstart': a, 'xend': b,\
                              'ystart': c, 'yend': d} for data, a, b, c, d in val_dicts]
        
        return val_dicts_array
        
    def train_mean_std(self):
        trainMean = self.trainMean
        trainStd = self.trainStd
        print(trainMean)
        print(trainStd)
        return trainMean, trainStd
