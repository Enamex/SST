import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description='video features to LSTM Language Model')

    # Location of data
    parser.add_argument('--dataset', type=str, default='ActivityNetCaptions',
                        help='Name of the data class to use from data.py')
    parser.add_argument('--data', type=str, default='data/ActivityNet/activity_net.v1-3.min.json',
                        help='location of the dataset')
    parser.add_argument('--features', type=str, default='data/ActivityNet/sub_activitynet_v1-3.c3d.hdf5',
                        help='location of the video features')
    parser.add_argument('--labels', type=str, default='data/ActivityNet/labels.hdf5',
                        help='location of the proposal labels')
    parser.add_argument('--vid-ids', type=str, default='data/ActivityNet/video_ids.json',
                        help='location of the video ids')

    parser.add_argument('--save', type=str, default='data/models/default',
                        help='path to folder where to save the final model and log files and corpus')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save the model every x epochs')
    parser.add_argument('--clean', dest='clean', action='store_true',
                        help='Delete the models and the log files in the folder')
    parser.add_argument('--W', type=int, default=128,
                        help='The rnn kernel size to use to get the proposal features')
    parser.add_argument('--K', type=int, default=64,
                        help='Number of proposals')
    parser.add_argument('--max-W', type=int, default=256,
                        help='maximum number of windows to return per video')

    # For ActiviyNet Captions
    parser.add_argument('--train-ids', type=str, default='data/ActivityNet/train_ids.json',
                        help='location of the train ids')
    parser.add_argument('--val-ids', type=str, default='data/ActivityNet/val_ids.json',
                        help='location of the val ids')
    parser.add_argument('--test-ids', type=str, default='data/ActivityNet/test_ids.json',
                        help='location of the test ids')
    parser.add_argument('--train', type=str, default='data/ActivityNet/train.json',
                        help='location of the training data')
    parser.add_argument('--val', type=str, default='data/ActivityNet/val_ids.json',
                        help='location of the validation data')
                        

    # Model options
    parser.add_argument('--rnn-type', type=str, default='GRU',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--rnn-num-layers', type=int, default=2,
                        help='Number of layers in rnn')
    parser.add_argument('--rnn-dropout', type=int, default=0.0,
                        help='dropout used in rnn')
    parser.add_argument('--video-dim', type=int, default=500,
                        help='dimensions of video (C3D) features')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='dimensions output layer of video network')

    # Training options
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout between RNN layers')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='SGD weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Print out debug sentences')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of training samples to train with')
    parser.add_argument('--shuffle', type=int, default=1,
                        help='whether to shuffle the data')
    parser.add_argument('--nthreads', type=int, default=1,
                        help='number of worker threas used to load data')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='reload the model')

    # Evaluate options
    parser.add_argument('--num-vids-eval', type=int, default=500,
                        help='Number of videos to evaluate at each pass')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='threshold above which we say something is positive')
    parser.add_argument('--num-proposals', type=int, default=None,
                        help='number of top proposals to evaluate')
    
    return parser.parse_args()