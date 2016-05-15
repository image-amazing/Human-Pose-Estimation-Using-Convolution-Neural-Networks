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
 
dofile 'Alexnet_PT/load_alexnet.lua'
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
--- START TRAINING
epochs = 200000;
batch_size = 200; --- spongebob
--learning_rate = 0.000003;
save_freq = 10
loss_array = torch.zeros(epochs)  --spongebob
Vloss_array = torch.zeros(epochs)

model:training()
dofile 'trainer.lua'
for k = 1,epochs do 
	train()
end

function hello()
	print('Hello world. Didn\'t expect me did ya? They call me the \'Torch\'erer. I shall haunt you in your dreams.');
end
hello();


