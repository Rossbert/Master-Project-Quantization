import numpy as np

batches = 1
size_y = 24
size_x = 24
out_channels = 32

x = []
for i in range(batches):
    for j in range(size_y):
        for k in range(size_x):
            for l in range(out_channels):
                x.append(i*size_y*size_x*out_channels + j*size_x*out_channels + k*out_channels + l)
#print(x)

position = 15270
position_limit = batches*size_y*size_x*out_channels
print(position_limit)
batch = int(position/(size_y*size_x*out_channels))
no_batch = position - batch*size_y*size_x*out_channels
ky = int(no_batch/(size_x*out_channels))
no_y = no_batch - ky*size_x*out_channels
kx = int(no_y/out_channels)
out_channel = no_y - kx*out_channels 
print(batch, ky, kx, out_channel)
print(batch*size_y*size_x*out_channels + ky*size_x*out_channels + kx*out_channels + out_channel)
#outputBatches * outputRows * outputCols * outputChannels;
#outputPosition = batch * outputRows * outputCols * outputChannels + output_y * outputCols * outputChannels + output_x * outputChannels + output_channel;

vals = []
acc = position
div = batches*size_y*size_x*out_channels
element = [batches, size_y, size_x, out_channels]

i = 0
while div > 1:
    div = div / element[i]
    vals.append(int(acc/div))
    acc = acc - vals[i]*div
    i = i + 1
    # print(div)
    # print(vals)
    # print(acc)
print(vals)
