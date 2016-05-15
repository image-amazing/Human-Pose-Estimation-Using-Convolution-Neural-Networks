--loads images, scales to 224x224, saves as t7. 
--saves original images x and y sizes as seperate file.


-- path './FLIC/images1'
require 'torch'
require 'xlua'
require 'image'
--pmaths=require 'pmaths'

----------------------------------------------------------------------
-- 1. Parse command-line arguments

op = xlua.OptionParser('load-images.lua [options]')
op:option{'-d', '--dir', action='store', dest='dir', help='directory to load', req=false}
op:option{'-e', '--ext', action='store', dest='ext', help='only load files of this extension', default='png'}
opt = op:parse()
op:summarize()

----------------------------------------------------------------------
-- 2. Load all files in directory

-- We process all files in the given dir, and add their full path
-- to a Lua table.

-- Create empty table to store file names:
files = {}
dir1 = 'lsp1_images_png'
dir2 = 'lsp2_images_png'
-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(dir1) do
   -- We only load files that match the extension
   if file:find(opt.ext .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(dir1,file))
   end
end


for file in paths.files(dir2) do
   -- We only load files that match the extension
   if file:find(opt.ext .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(dir2,file))
   end
end

-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ' .. opt.ext)
end

----------------------------------------------------------------------
-- 3. Sort file names

-- We sort files alphabetically, it's quite simple with table.sort()
table.sort(files, function (a,b) return a < b end)
print(files)

print(table.getn(files))

torch.save('lsp_table.t7',files)
--[[
----------------------------------------------------------------------
-- 4. Finally we load images

-- Go over the file list:
--images = {}
local n = table.getn(files);
local w=224;
local h=224;
images_tensor = torch.Tensor(table.getn(files)/2,3,h,w);
original_sizes_tensor = torch.Tensor(table.getn(files),2);
for i=1,n/2 do
   -- load each image
   --table.insert(images, image.load(file))
   -- load as a tensor
   print(files[i])
   photo = image.load(files[i]);
   --print(#photo)
   
   original_sizes_tensor[{{i},{}}] = torch.Tensor({{photo:size(2),photo:size(3)}})--2 is height 3 is width
   images_tensor[{{i},{},{},{}}] = image.scale(photo,h,w);
   photo =nil
   collectgarbage()
  
end

print('Loaded images:')
print(#images_tensor)
print('saving')
torch.save('lsp_images_train1.t7',images_tensor[{{},{},{},{}}]) --6000



images_tensor = nil
collectgarbage()

images_tensor = torch.Tensor(table.getn(files)/2,3,h,w);

for i=(n/2)+1,n do
   -- load each image
   --table.insert(images, image.load(file))
   -- load as a tensor
    print(files[i])
    photo = image.load(files[i]);
     --print(#photo)
     original_sizes_tensor[{{i},{}}] = torch.Tensor({{photo:size(2),photo:size(3)}})--2 is height 3 is width
     images_tensor[{{i-n/2},{},{},{}}] = image.scale(photo,h,w);

     photo = nil
     collectgarbage()
   end
   --

print('Loaded images:')
print(#images_tensor)
print('saving')
torch.save('lsp_images_train2.t7',images_tensor[{{1,5000},{},{},{}}])
torch.save('lsp_images_test.t7',images_tensor[{{5001,6000},{},{},{}}])
torch.save('lsp_images_size_train.t7',original_sizes_tensor[{{1,11000},{}}])
torch.save('lsp_images_size_test.t7',original_sizes_tensor[{{11001,12000},{}}])
print('done')


-- Display a of few them
--for i = 1,math.min(#files,10) do
  -- image.display{image=images[i], legend=files[i]}
--end

-- in qlua 
--images = torch.load('lsp_images1.t7')
--image.display{image=images[{{i},{},{},{}}]}
--
--]]
