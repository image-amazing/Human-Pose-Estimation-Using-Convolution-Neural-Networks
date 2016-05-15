require 'cudnn'
--require 'cutorch'
local nn= require 'nn'
local loadcaffe = require 'loadcaffe'

local dir = '/scratch/eecs442w16_fluxg/rounak/cv_project/Alexnet_PT/'
local proto = 'deploy.prototxt'
local caffemodel = 'bvlc_alexnet.caffemodel'

model = loadcaffe.load(dir..proto,dir..caffemodel,'cudnn')
print(model)
--
--
--Modify last layer
model:remove(#model)
model:remove(#model)
model:add(nn.Linear(4096,28))


--Glorot type Initialization 
local function glorot_init(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end


for i,mod in ipairs(model.modules) do
	
	if torch.type(mod):find('Conv')  then
		mod.accGradParameters = function() end
		mod.updateParameters = function() end
		print('layer #' .. i,'\t freeze conv')
	elseif torch.type(mod):find('Linear') then
		mod:reset(glorot_init(mod.weight:size(2), mod.weight:size(1)))
		print('layer #' .. i,'\treinitialize linear')
		
		if  mod.bias then
        	mod.bias:zero()
		end
	end

end

