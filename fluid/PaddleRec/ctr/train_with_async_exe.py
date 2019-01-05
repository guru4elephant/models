import argparse
import logging
import os
import paddle
import paddle.fluid as fluid
from nets import bow_net

def parse_args():
    parser = argparse.ArgumentParser(description="Tower CTR example")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="The size of mini-batch (default:32)")
    parser.add_argument(
        '--thread_num',
        type=int,
        default=10,
        help="thread num")
    parser.add_argument(
        '--num_passes',
        type=int,
        default=10,
        help="The number of passes to train (default: 10)")
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default='models',
        help='The path for model to store (default: models)')

    return parser.parse_args()

class Model(object):
    def __init__(self):
        self.label = fluid.layers.data(name="slot_0", shape=[-1, 1],
                                       dtype="int64", lod_level=0,
                                       append_batch_size=False)
        self.user = []
        self.item = []
        for i in range(1, 70, 1):
            self.user.append(fluid.layers.data(name="slot_%d" % i, shape=[1],
                                               dtype="int64", lod_level=1))
        for j in range(70, 100, 1):
            self.item.append(fluid.layers.data(name="slot_%d" % i, shape=[1],
                                               dtype="int64", lod_level=1))
        self.avg_cost, prediction = bow_net(self.user, self.item, self.label)


def async_train(args):
    model = Model()
    optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
    optimizer.minimize(model.avg_cost)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    data_set = data_feed.DataFeedDesc("ctr.proto")
    input_data_name = [model.label] + model.user + model.item
    data_set.set_use_slots(input_data_name)
    data_set.set_batch_size(args.batch_size)
    async_exe = fluid.AsyncExecutor()
    file_list = ["./data/part-%s" % str(i).zfill(5) for i in range(100)]
    epoch_num = args.num_passes
    thread = args.thread_num
    model_dir = args.model_output_dir
    if not os.path.iddir(args.model_output_dir):
        os.mkdir(args.model_otuput_dir)

    for i in range(epoch_num):
        async_exe.run(fluid.default_main_program(), data_set,
                      file_list, thread, loss, debug=True)
        fluid.io.save_inference_model(model_dir, input_data_name, [loss], exe)

if __name__ == '__main__':
    args = parse_args()
    async_train(args)
