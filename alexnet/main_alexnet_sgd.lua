-- Run Before: flux_load-images.lua to create lsp_images_size.t7 and lsp_images.t7
-- Depends on: alexnet4.lua
-- Need these files: lsp_joints.t7

require('nn')
require('optim')
require('cunn')
require('cutorch')
require('loadcaffe')
require('cudnn')
cutorch.setDevice(1)

h=224
 loaded_chunk = assert(loadfile("alexnet.lua"))
loaded_chunk();
net = create_net();
model = net.model;
--VGG
--[[
model = loadcaffe.load('vgg_PT/deploy.prototxt','vgg_PT/VGG.caffemodel');
model:remove(40)
model:remove(39)
model:remove(36)
model:remove(33)
model:insert(nn.Linear(25088,4096),33)
model:insert(nn.Linear(4096,4096),36)
model:add(nn.Linear(4096,1000))
model:add(nn.Linear(1000, 28))
--]]
net = nil
collectgarbage()

--dofile 'NIN_PT/load_NIN.lua'
print(model)

model:cuda()
criterion = nn.MSECriterion():cuda(); -----------------------------------------spongebob


-- example accessing layer:
-- alexnet.model.modules[1].modules[10]

--SANITY CHECK
do
	local input = torch.Tensor(3,h,h):cuda();
	local output = model:forward(input);
	print('output size')
	print(#output)
	local gradInput = model:backward(input, torch.CudaTensor(28))
	criterion:forward(output, torch.CudaTensor(28))
end
--


--CREATE trainset
labels = torch.load('lsp_joints_train.t7'); -- 
 
labels = labels[{{1,2},{}}] --2x14xn

--Scale labels in accordance to image scaling
do
local original_image_size = torch.load('lsp_images_size_train.t7');
--trainset_size = original_image_size:size(1)
trainset_size = 11000  -------------------------------------------------------------------------------------- spongebob
	for i=1,original_image_size:size(1) do 
		labels[{{1},{},{i}}]:div(original_image_size[{{i},{2}}]:squeeze() /h)
		labels[{{2},{},{i}}]:div(original_image_size[{{i},{1}}]:squeeze() /h)
	end

labels = labels:reshape(28,original_image_size:size(1)):transpose(1,2) --28xn -- (x1,x2,x3...x14,y1,y2,y3...y14)'
end

--print(#labels)
trainset = {labels = labels}
--print(torch.type(trainset.data))

-- CREATE testset

labels = torch.load('lsp_joints_test.t7'); --
--labels1 = labels[{{1,2},{},{}}]:clone() -- 2x14xn
labels = labels[{{1,2},{}}]

local original_image_size = torch.load('lsp_images_size_test.t7');
--testset_size = original_image_size:size(1)
testset_size = 200  -----------------------------------------------------------------------------------------spongebob
        for i=1,original_image_size:size(1) do
                labels[{{1},{},{i}}]:div(original_image_size[{{i},{2}}]:squeeze() /h)
                labels[{{2},{},{i}}]:div(original_image_size[{{i},{1}}]:squeeze() /h)
        end

labels = labels:reshape(28,original_image_size:size(1)):transpose(1,2) --28xn -- (x1,x2,x3...x14,y1,y2,y3...y14)'

testset = {data,labels = labels[{{1,testset_size}}]}
--]]
--[[
--PREPROCESS DATA
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
torch.save('mean.t7',mean)
torch.save('stdv.t7',stdv)

testBatch = trainset.data[{{trainset_size+1,trainset_size+20}}]
testBatchLabels = trainset.labels[{{trainset_size+1,trainset_size+20}}] ------------------------spongebob
trainset.data = trainset.data[{{1,trainset_size}}]
trainset.labels = trainset.labels[{{1,trainset_size}}]
--]]
-- START TRAINING
epochs = 2000;
batch_size = 200; --- spongebob
--learning_rate = 0.000003;
save_freq = 10
loss_array = torch.zeros(epochs)  --spongebob
Vloss_array = torch.zeros(epochs)

inputs = torch.Tensor(batch_size,3,h,h)
--inputs = inputs:cuda()
outputs = torch.Tensor(batch_size,28)
outputs:cuda()
for i = 1,epochs do

	for j = 1,batch_size do
		x = torch.random(1,6000)
		inputs[j] = trainset.data[x]
		outputs[j] = trainset.labels[x]
		
	end
--forward pass
	inputs = inputs:cuda()
	model_out = model:forward(inputs):cuda()
	loss = criterion:forward(model_out, outputs)
	loss_array[i] = loss
--
--
	print(model_out[{{1},{}}])
--print loss
	print('epoch'..i..'     loss'..loss)
--
-- Save loss,output labels
--io.write("Loss = ",loss,"\n")
	if ((i/save_freq-math.floor(i/save_freq)) == 0) then
		torch.save('./loss/loss'..i..'.t7', loss)
d	torch.save('./model/model'..i..'.t7', model)
		torch.save('./model_out/model_out'..i..'.t7',model_out)
		torch.save('./loss_array/loss_array'..i..'.t7',loss_array)
	end



--Zero the accumulation of gradients
	model:zeroGradParameters()
--
--back prop
	model:backward(inputs, criterion:backward(model_out, outputs))
--
--update params

	model:updateParameters(learning_rate)
 --
end




--]]
