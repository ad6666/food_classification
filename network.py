from mindspore import nn
import mindspore as ms
import numpy as np
def network():
    network = nn.SequentialCell([
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), pad_mode='pad', padding=3),
        nn.BatchNorm2d(num_features=64,eps=1e-05,momentum=0.09999999999999998),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='SAME'),

        nn.SequentialCell([
            nn.SequentialCell([
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad", padding=1),
                nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU(),

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU()
            ]),

            nn.SequentialCell([
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU(),

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU()
            ])
        ]),
        nn.SequentialCell([
            nn.SequentialCell([
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU(),

                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU()
            ]),

            nn.SequentialCell([
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU(),

                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode="pad",
                          padding=1),
                nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.09999999999999998),
                nn.ReLU()
            ])
        ]),
        nn.AvgPool2d(kernel_size=3, stride=3),
        nn.Flatten(),
        nn.Dropout(0.6),
        nn.Dense(128 * 10 * 10, 10)
    ])
    return network
if __name__ == '__main__':
    inputs = ms.Tensor(np.ones([1, 3, 128, 128]).astype(np.float32))
    print(network()(inputs).shape)


# C:\Users\23670\AppData\Local\Programs\Python\Python39\python.exe C:\Users\23670\Desktop\python\深度学习\food\main.py
# BaseClassifier<
#   (backbone): ResNet<
#     (conv1): ConvNormActivation<
#       (features): SequentialCell<
#         (0): Conv2d<input_channels=3, output_channels=64, kernel_size=(7, 7), stride=(2, 2), pad_mode=pad, padding=3, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#         (1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.conv1.features.1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.conv1.features.1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.conv1.features.1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.conv1.features.1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
#         (2): ReLU<>
#         >
#       >
#     (max_pool): MaxPool2d<kernel_size=3, stride=2, pad_mode=SAME>
#     (layer1): SequentialCell<
#       (0): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer1.0.conv1.features.1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer1.0.conv1.features.1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer1.0.conv1.features.1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer1.0.conv1.features.1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer1.0.conv2.features.1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer1.0.conv2.features.1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer1.0.conv2.features.1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer1.0.conv2.features.1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         >
#       (1): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer1.1.conv1.features.1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer1.1.conv1.features.1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer1.1.conv1.features.1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer1.1.conv1.features.1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer1.1.conv2.features.1.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer1.1.conv2.features.1.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer1.1.conv2.features.1.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer1.1.conv2.features.1.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         >
#       >
#     (layer2): SequentialCell<
#       (0): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=64, output_channels=128, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer2.0.conv1.features.1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer2.0.conv1.features.1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer2.0.conv1.features.1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer2.0.conv1.features.1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer2.0.conv2.features.1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer2.0.conv2.features.1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer2.0.conv2.features.1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer2.0.conv2.features.1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         (down_sample): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=64, output_channels=128, kernel_size=(1, 1), stride=(2, 2), pad_mode=pad, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer2.0.down_sample.features.1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer2.0.down_sample.features.1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer2.0.down_sample.features.1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer2.0.down_sample.features.1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         >
#       (1): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer2.1.conv1.features.1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer2.1.conv1.features.1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer2.1.conv1.features.1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer2.1.conv1.features.1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=128, output_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=128, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer2.1.conv2.features.1.gamma, shape=(128,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer2.1.conv2.features.1.beta, shape=(128,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer2.1.conv2.features.1.moving_mean, shape=(128,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer2.1.conv2.features.1.moving_variance, shape=(128,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         >
#       >
#     (layer3): SequentialCell<
#       (0): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=128, output_channels=256, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer3.0.conv1.features.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer3.0.conv1.features.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer3.0.conv1.features.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer3.0.conv1.features.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer3.0.conv2.features.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer3.0.conv2.features.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer3.0.conv2.features.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer3.0.conv2.features.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         (down_sample): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=128, output_channels=256, kernel_size=(1, 1), stride=(2, 2), pad_mode=pad, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer3.0.down_sample.features.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer3.0.down_sample.features.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer3.0.down_sample.features.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer3.0.down_sample.features.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         >
#       (1): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer3.1.conv1.features.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer3.1.conv1.features.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer3.1.conv1.features.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer3.1.conv1.features.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=256, output_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=256, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer3.1.conv2.features.1.gamma, shape=(256,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer3.1.conv2.features.1.beta, shape=(256,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer3.1.conv2.features.1.moving_mean, shape=(256,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer3.1.conv2.features.1.moving_variance, shape=(256,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         >
#       >
#     (layer4): SequentialCell<
#       (0): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=256, output_channels=512, kernel_size=(3, 3), stride=(2, 2), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer4.0.conv1.features.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer4.0.conv1.features.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer4.0.conv1.features.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer4.0.conv1.features.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer4.0.conv2.features.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer4.0.conv2.features.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer4.0.conv2.features.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer4.0.conv2.features.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         (down_sample): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=256, output_channels=512, kernel_size=(1, 1), stride=(2, 2), pad_mode=pad, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer4.0.down_sample.features.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer4.0.down_sample.features.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer4.0.down_sample.features.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer4.0.down_sample.features.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         >
#       (1): ResidualBlockBase<
#         (conv1): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer4.1.conv1.features.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer4.1.conv1.features.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer4.1.conv1.features.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer4.1.conv1.features.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
#             (2): ReLU<>
#             >
#           >
#         (conv2): ConvNormActivation<
#           (features): SequentialCell<
#             (0): Conv2d<input_channels=512, output_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode=pad, padding=1, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#             (1): BatchNorm2d<num_features=512, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=backbone.layer4.1.conv2.features.1.gamma, shape=(512,), dtype=Float32, requires_grad=True), beta=Parameter (name=backbone.layer4.1.conv2.features.1.beta, shape=(512,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=backbone.layer4.1.conv2.features.1.moving_mean, shape=(512,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=backbone.layer4.1.conv2.features.1.moving_variance, shape=(512,), dtype=Float32, requires_grad=False)>
#             >
#           >
#         (relu): ReLU<>
#         >
#       >
#     >
#   (neck): GlobalAvgPooling<>
#   (head): DenseHead<
#     (dropout): Dropout<keep_prob=1.0>
#     (dense): Dense<input_channels=512, output_channels=1000, has_bias=True>
#     >
#   >
#
# 进程已结束,退出代码0
