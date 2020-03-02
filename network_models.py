from keras.layers import (Reshape, Dropout, Flatten, Conv1D, Permute, LSTM, Conv3D, MaxPool3D, Lambda, MaxPool1D,
                          Concatenate, Add, BatchNormalization, Activation, SpatialDropout3D, SpatialDropout1D)
from keras.models import Model, Input
from keras import backend as K


def _reshape_time_features(net):
    '''Reshape video features in a format compatible with LSTM layer'''
    _, ny, nx, nt, nft = net._shape_tuple()
    net = Permute((3, 1, 2, 4))(net)
    net = Reshape((nt, nx*ny*nft))(net)
    return net


def create_video_encoder(name, frame_size=(100, 100, 100), use_bn=False, use_dropout=True):
    '''Create the video encoder branch, or f_sil()'''
    # Input
    inputs = Input(shape=(frame_size[0], frame_size[1], frame_size[2]))

    # CNN
    net = Reshape((frame_size[0], frame_size[1], frame_size[2], 1))(inputs)

    t_pool = (2, 2, 2)
    t_same = (2, 2, 1)
    if name == 'LSTM_with_pool':
        net = Conv3D(8, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='same')(net)

        net = Conv3D(16, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = Conv3D(32, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='same')(net)

        net = Conv3D(64, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = Conv3D(128, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='same')(net)

        net = _reshape_time_features(net)
        net = Dropout(0.2)(net) if use_dropout else net
        encoded = LSTM(128, recurrent_dropout=0.1)(net) if use_dropout else LSTM(128)
    elif name == 'LSTM_no_pool':
        net = Conv3D(8, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = Conv3D(16, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = Conv3D(32, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = Conv3D(64, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = Conv3D(128, kernel_size=3, padding='same')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_same, padding='same')(net)

        net = _reshape_time_features(net)
        net = Dropout(0.2)(net) if use_dropout else net
        encoded = LSTM(128, recurrent_dropout=0.1)(net) if use_dropout else LSTM(128)
    elif name == 'fully_conv':
        net = Conv3D(16, kernel_size=3, padding='valid')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='valid')(net)

        net = Conv3D(32, kernel_size=3, padding='valid')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='valid')(net)

        net = Conv3D(64, kernel_size=3, padding='valid')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv3D(64, kernel_size=3, padding='valid')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='valid')(net)

        net = Conv3D(128, kernel_size=3, padding='valid')(net)
        net = SpatialDropout3D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool3D(pool_size=t_pool, padding='valid')(net)

        net = Dropout(0.2)(net) if use_dropout else net
        net =  Conv3D(128, kernel_size=3, padding='valid')(net)
        net = Activation('tanh')(net)
        encoded = Flatten()(net)

    model = Model(inputs=inputs, outputs=encoded)
    model.summary()

    return model


def create_boxacc_encoder(name, vec_size, use_bn=False, use_dropout=True):
    '''Create the accelerometer encoder branch, f_bb() and g()'''
    # Input
    inputs = Input(shape=(vec_size[0], vec_size[1]))

    t_pool = 2
    t_same = 1
    if name == 'LSTM_with_pool':
        net = Conv1D(8, kernel_size=3, padding='same')(inputs)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='same')(net)

        net = Conv1D(16, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(32, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='same')(net)

        net = Conv1D(64, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(128, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='same')(net)

        net = Dropout(0.2)(net) if use_dropout else net
        encoded = LSTM(128, recurrent_dropout=0.1)(net) if use_dropout else LSTM(128)
    elif name == 'LSTM_no_pool':
        net = Conv1D(8, kernel_size=3, padding='same')(inputs)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(16, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(32, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(64, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(128, kernel_size=3, padding='same')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Dropout(0.2)(net) if use_dropout else net
        encoded = LSTM(128, recurrent_dropout=0.1)(net) if use_dropout else LSTM(128)
    elif name == 'fully_conv':
        net = Conv1D(16, kernel_size=3, padding='valid')(inputs)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='valid')(net)

        net = Conv1D(32, kernel_size=3, padding='valid')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='valid')(net)

        net = Conv1D(64, kernel_size=3, padding='valid')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)

        net = Conv1D(64, kernel_size=3, padding='valid')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='valid')(net)

        net = Conv1D(128, kernel_size=3, padding='valid')(net)
        net = SpatialDropout1D(0.2)(net) if use_dropout else net
        net = BatchNormalization()(net) if use_bn else net
        net = Activation('relu')(net)
        net = MaxPool1D(pool_size=t_pool, padding='valid')(net)

        net = Dropout(0.2)(net) if use_dropout else net
        net =  Conv1D(128, kernel_size=3, padding='valid')(net)
        net = Activation('tanh')(net)
        encoded = Flatten()(net)

    model = Model(inputs=inputs, outputs=encoded)
    model.summary()

    return model


def triplet_network(name, frame_size=(100, 100, 100), box_size=(100, 4), acc_size=(100, 3), use_bn=False, use_dropout=True):
    '''Create the triplet network combining the video encoder and the accelerometer encoder'''
    input_vid = Input(shape=frame_size)
    input_box = Input(shape=box_size)
    input_acc_pos = Input(shape=acc_size)
    input_acc_neg = Input(shape=acc_size)

    vid_encoder = create_video_encoder(name, frame_size, use_bn, use_dropout)
    box_encoder = create_boxacc_encoder(name, box_size, use_bn, use_dropout)
    acc_encoder = create_boxacc_encoder(name, acc_size, use_bn, use_dropout)

    encoded_vid = vid_encoder(input_vid)
    encoded_box = box_encoder(input_box)
    encoded_anchor = Add()([encoded_vid, encoded_box])
    encoded_acc_pos = acc_encoder(input_acc_pos)
    encoded_acc_neg = acc_encoder(input_acc_neg)

    distance_good = Lambda(euclidean_distance)([encoded_anchor, encoded_acc_pos])
    distance_bad = Lambda(euclidean_distance)([encoded_anchor, encoded_acc_neg])
    distances = Concatenate(axis=1)([distance_good, distance_bad])

    model = Model([input_vid, input_box, input_acc_pos, input_acc_neg], distances)
    model.summary()

    return model


def triplet_loss(y_true, y_pred):
    '''Standard Triplet loss'''
    distance_good = y_pred[:, 0]
    distance_bad = y_pred[:, 1]
    margin = 0.2
    loss = distance_good - distance_bad + margin
    return K.maximum(loss, 0.0)


def rtl_loss(y_true, y_pred):
    '''Reciprocal Triplet Loss'''
    distance_good = y_pred[:, 0]
    distance_bad = y_pred[:, 1]
    loss = (distance_good) + 1/(distance_bad + K.epsilon())
    return loss


def triplet_acc(y_true, y_pred):
    '''Proxy for the triplet accuracy estimated per batch'''
    distance_good = y_pred[:, 0]
    distance_bad = y_pred[:, 1]
    return K.cast(K.less_equal(distance_good, distance_bad), K.floatx())


def good_distance(y_true, y_pred):
    '''Function used to monitor the matching distance during training'''
    distance_good = y_pred[:, 0]
    return K.mean(distance_good)


def bad_distance(y_true, y_pred):
    '''Function used to monitor the non-matching distance during training'''
    distance_bad = y_pred[:, 1]
    return K.mean(distance_bad)


def euclidean_distance(vects):
    '''Calculate the Euclidean distance between two tensors'''
    x, y = vects
    distance = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.maximum(distance, K.epsilon())


if __name__ == '__main__':
    siamese = triplet_network('fully_conv', frame_size=(100, 100, 100), box_size=(100, 4), acc_size=(100, 3), use_bn=True)
