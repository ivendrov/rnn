------------------------------------------------------------------------
--[[ GenLSTM ]]--
-- Long Short Term Memory architecture, with arbitrary nonlinearities for
-- the "block input" and "block output" activations
-- (g and h in http://arxiv.org/pdf/1503.04069v1.pdf)
-- Otherwise identical to LSTM.lua
------------------------------------------------------------------------
local GenLSTM, parent = torch.class('nn.GenLSTM', 'nn.LSTM')

function GenLSTM:__init(inputSize, outputSize, g, h, rho)
    parent.__init(self, inputSize, outputSize, rho)

    self.g = g or nn.Tanh
    self.h = h or nn.Tanh
end


function GenLSTM:buildHidden()
    local hidden = parent.buildHidden(self)
    hidden:remove(#hidden.modules)
    hidden:add(self.g())
    return hidden
end


function GenLSTM:buildModel()
   -- build components
   self.cellLayer = self:buildCell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   local concat2 = nn.ConcatTable()
   concat2:add(nn.SelectTable(1)):add(nn.SelectTable(2))
   concat:add(concat2):add(self.cellLayer)
   local model = nn.Sequential()
   model:add(concat)
   -- output of concat is {{input, output}, cell(t)},
   -- so flatten to {input, output, cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(self.h())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end