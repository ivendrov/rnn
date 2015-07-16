------------------------------------------------------------------------
--[[ MultiLayerRecurrent ]]--
-- Container module for multiple stacked recurrent layers
-- (the output of each layer is fed into the input of the next)
--
-- This module is useful when you need to be able to input the sequence
-- one step at a time; otherwise, nn.Sequential and Sequencer-wrapped recurrent modules
-- work just as well.
--
-- All submodules must have the same rho (maximum number of time steps for BPTT)
------------------------------------------------------------------------

-- not subclass of AbstractRecurrent because many of AbstractRecurrent methods
-- (e.g. getStepModule, includingSharedClones) don't make sense for this class.
local MultiLayerRecurrent, parent = torch.class('nn.MultiLayerRecurrent', 'nn.Sequential')

function MultiLayerRecurrent:__init(rho)
    parent.__init(self)
    self.rho = rho
end


function MultiLayerRecurrent:add(recurrentModule)
    assert(recurrentModule.rho == self.rho, 'Component module has different rho from parent')
    parent.add(self, recurrentModule)
end

function MultiLayerRecurrent:updateGradInput(input, gradOutput)
   -- just call this for the top layer; we don't know what the gradOutput for the bottom layers
   -- are until we do BPTT
   local top_module = self.modules[#self.modules]
   top_module:updateGradInput(input, gradOutput)
end

function MultiLayerRecurrent:accGradParameters(input, gradOutput, scale)
   -- set the scales for every layer
   for _, module in ipairs(self.modules) do
        module.scales[module.step-1] = scale or 1
   end
end

function MultiLayerRecurrent:backwardThroughTime()
    local n = #self.modules

    -- backward pass over modules in reverse order
    local cur_module = self.modules[n]
    local lastGradInput = cur_module:backwardThroughTime()
    for i = 1, n-1 do
        local next_module = self.modules[n - i]
        next_module.gradOutputs = cur_module.gradInputs
        lastGradInput = next_module:backwardThroughTime()
        cur_module = next_module
    end

    return lastGradInput
end

function MultiLayerRecurrent:forget(offset)
    for _, module in ipairs(self.modules) do
        module:forget(offset)
    end
end

function MultiLayerRecurrent:recycle(offset)
    for _, module in ipairs(self.modules) do
        module:recycle(offset)
    end
end





