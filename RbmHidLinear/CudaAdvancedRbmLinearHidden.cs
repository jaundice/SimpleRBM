using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using Cudafy.Host;
using Cudafy.Maths.RAND;
/*
http://www.cs.toronto.edu/~hinton/code/rbmhidlinear.m

% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, tochastic real-valued feature detectors drawn from a unit
% variance Gaussian whose mean is determined by the input from 
% the logistic visible units. Learning is done with 1-step Contrastive Divergence.
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning

epsilonw      = 0.001; % Learning rate for weights 
epsilonvb     = 0.001; % Learning rate for biases of visible units
epsilonhb     = 0.001; % Learning rate for biases of hidden units 
weightcost  = 0.0002;  
initialmomentum  = 0.5;
finalmomentum    = 0.9;


[numcases numdims numbatches]=size(batchdata);

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases.
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);


  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  sigmainc = zeros(1,numhid);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch,
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;

 for batch = 1:numbatches,
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs =  (data*vishid) + repmat(hidbiases,numcases,1);
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);
  
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
poshidstates = poshidprobs+randn(numcases,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  neghidprobs = (negdata*vishid) + repmat(hidbiases,numcases,1);
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  err= sum(sum( (data-negdata).^2 )); 
  errsum = err + errsum;
   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end
fprintf(1, 'epoch %4i error %f \n', epoch, errsum);

end

*/
using SimpleRBM.Common;
using SimpleRBM.Cuda;
#if USEFLOAT
using TElement = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;
#else
using TElement = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
#endif

namespace CudaNN
{
    [Serializable]
    public class CudaAdvancedRbmLinearHidden : CudaAdvancedRbmBase
    {
        private TElement _trainingRandStDev;

        public CudaAdvancedRbmLinearHidden(GPGPU gpu, GPGPURAND rand, int layerIndex, int numVisibleNeurons,
            int numHiddenNeurons,
            TElement weightcost = (TElement) 0.0002,
            TElement initialMomentum = (TElement) 0.5, TElement finalMomentum = (TElement) 0.9, TElement weightInitializationStDev = (TElement)0.01, TElement trainRandStDev = (TElement)0.5, TElement momentumIncrementStep = 0.01)
            : base(
                gpu, rand, layerIndex, numVisibleNeurons, numHiddenNeurons, /*epsilonw, epsilonvb, epsilonhb,*/
                weightcost, initialMomentum, finalMomentum, weightInitializationStDev, momentumIncrementStep)
        {
            _trainingRandStDev = trainRandStDev;
        }

        protected override void LoadSpecific(SerializationInfo info)
        {
            base.LoadSpecific(info);
#if USEFLOAT 
            _trainingRandStDev = info.GetSingle("trainStDev");
#else
            _trainingRandStDev = info.GetDouble("trainStDev");
#endif
        }

        protected override void SaveSpecific(SerializationInfo info)
        {
            base.SaveSpecific(info);
            info.AddValue("trainStDev", _trainingRandStDev);
        }

        public override Matrix2D<TElement> Encode(Matrix2D<TElement> data)
        {
            var state = State;
            Wake();

            var numcases = data.GetLength(0);

            using (Matrix2D<TElement> tiledHiddenBiases = AsCuda.HiddenBiases.RepMatRows(numcases))
            using (Matrix2D<TElement> datavishid = data.Multiply(AsCuda.Weights))
            {
                SetState(state);
                return datavishid.Add(tiledHiddenBiases);
            }
        }

        public override Matrix2D<TElement> Decode(Matrix2D<TElement> activations)
        {
            var state = State;
            Wake();

            var numcases = activations.GetLength(0);

            Matrix2D<TElement> negdata;
            using (Matrix2D<TElement> tiledvisBiases = AsCuda.VisibleBiases.RepMatRows(numcases))
            using (Matrix2D<TElement> vishidtransposed = AsCuda.Weights.Transpose())
            {
                negdata = activations.Multiply(vishidtransposed);
                negdata.SubtractInPlace(tiledvisBiases);
                negdata.LogisticInPlace();
            }

            SetState(state);
            return negdata;
        }


        protected override TElement BatchedTrainEpoch(Matrix2D<TElement> data, Matrix2D<TElement> dataTransposed,
            Matrix2D<TElement> posvisact, int epoch, int numcases,
            TElement weightLearningRate,
            TElement hidBiasLearningRate,
            TElement visBiasLearningRate,
            TElement momentum)
        {
            TElement error;
            //start positive phase
            int batchCases = data.GetLength(0);
            //numcases = batchCases;//testing

            using (Matrix2D<TElement> tiledHiddenBiases = AsCuda.HiddenBiases.RepMatRows(batchCases))
            using (Matrix2D<TElement> datavishid = data.Multiply(AsCuda.Weights))
            using (Matrix2D<TElement> poshidprobs = datavishid.Add(tiledHiddenBiases))
            using (Matrix2D<TElement> posprobs = dataTransposed.Multiply(poshidprobs))
            using (Matrix2D<TElement> poshidact = poshidprobs.SumColumns())
            {
                //end of positive phase
                Matrix2D<TElement> poshidstates;
                using (
                    Matrix2D<TElement> rand = AsCuda.GPU.GuassianDistribution(AsCuda.GPURAND, batchCases,
                        NumHiddenNeurons,
                        scale: (TElement)1, stDev: _trainingRandStDev))
                {
                    poshidstates = poshidprobs.Add(rand);
                }

                //start neg phase
                Matrix2D<TElement> negdata;
                using (poshidstates)
                using (Matrix2D<TElement> tiledvisBiases = AsCuda.VisibleBiases.RepMatRows(batchCases))
                using (Matrix2D<TElement> vishidtransposed = AsCuda.Weights.Transpose())
                {
                    negdata = poshidstates.Multiply(vishidtransposed);
                    negdata.SubtractInPlace(tiledvisBiases);
                    negdata.LogisticInPlace();
                }

                Matrix2D<TElement> neghidact;
                Matrix2D<TElement> negvisact;
                Matrix2D<TElement> negprods;

                using (Matrix2D<TElement> neghidprobs = negdata.Multiply(AsCuda.Weights))
                using (Matrix2D<TElement> negdatatransposed = negdata.Transpose())
                {
                    neghidprobs.AddInPlace(tiledHiddenBiases);

                    negprods = negdatatransposed.Multiply(neghidprobs);
                    neghidact = neghidprobs.SumColumns();
                    negvisact = negdata.SumColumns();
                }


                //end of neg phase
                using (negdata)
                using (Matrix2D<TElement> delta = data.Subtract(negdata))
                {
                    delta.PowInPlace(2);
                    //using (Matrix2D<TElement> errcols = delta.SumColumns())
                    //using (Matrix2D<TElement> errrows = errcols.SumRows())
                    //{
                    //    error = errrows.CopyLocal()[0, 0];
                    //}
                    error = delta.Sum();
                }

                using (negprods)
                using (Matrix2D<TElement> momentumvishidinc = WeightInc.Multiply(momentum))
                //using (Matrix2D<TElement> posprodsminusnegprods = posprobs.Subtract(negprods).MultiplyInPlace((TElement)1 / (TElement)numcases))
                using (Matrix2D<TElement> posprodsminusnegprods = posprobs.Subtract(negprods).MultiplyInPlace((TElement)1 / (TElement)batchCases))
                using (Matrix2D<TElement> vishidweightcost = AsCuda.Weights.Multiply(WeightCost))
                {
                    posprodsminusnegprods.SubtractInPlace(vishidweightcost);
                    posprodsminusnegprods.MultiplyInPlace(weightLearningRate);

                    WeightInc.Dispose();
                    _vishidinc = momentumvishidinc.Add(posprodsminusnegprods);
                }

                using (negvisact)
                using (Matrix2D<TElement> momentumvisbiasinc = VisibleBiasInc.Multiply(momentum))
                //using (Matrix2D<TElement> posvisactminusnegvisact = posvisact.Subtract(negvisact).MultiplyInPlace(visBiasLearningRate / (TElement)numcases))
                using (Matrix2D<TElement> posvisactminusnegvisact = posvisact.Subtract(negvisact).MultiplyInPlace(visBiasLearningRate / (TElement)batchCases))
                {
                    VisibleBiasInc.Dispose();
                    _visbiasinc = momentumvisbiasinc.Add(posvisactminusnegvisact);
                }
                using (neghidact)
                using (Matrix2D<TElement> momentumhidbiasinc = HiddenBiasInc.Multiply(momentum))
                //using (Matrix2D<TElement> poshidactminusneghidact = poshidact.Subtract(neghidact).MultiplyInPlace(hidBiasLearningRate / (TElement)numcases))
                using (Matrix2D<TElement> poshidactminusneghidact = poshidact.Subtract(neghidact).MultiplyInPlace(hidBiasLearningRate / (TElement)batchCases))
                {
                    HiddenBiasInc.Dispose();
                    _hidbiasinc = momentumhidbiasinc.Add(poshidactminusneghidact);
                }

                AsCuda.Weights.AddInPlace(WeightInc);
                AsCuda.VisibleBiases.AddInPlace(VisibleBiasInc);
                AsCuda.HiddenBiases.AddInPlace(HiddenBiasInc);
            }

            return error;
        }


    }
}