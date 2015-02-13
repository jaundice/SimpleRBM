using System.Diagnostics;
using Cudafy.Host;
using Cudafy.Maths.RAND;
using SimpleRBM.Common;
using SimpleRBM.Cuda;
/*
http://www.cs.toronto.edu/~hinton/code/rbm.m
% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
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
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
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
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch,
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
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
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end;
*/
#if USEFLOAT
using TElementType = System.Single;
using xxx = SimpleRBM.Cuda.CudaRbmF;

#else
using TElementType = System.Double;
using xxx = SimpleRBM.Cuda.CudaRbmD;
#endif

namespace CudaNN
{
    public class CudaAdvancedRbmBinary : CudaAdvancedRbmBase
    {
        public bool ConvertActivationsToStates { get; protected set; }

        private TElementType _decodingNoiseLevel;
        private TElementType _encodingNoiseLevel;

        public CudaAdvancedRbmBinary(GPGPU gpu, GPGPURAND rand, int layerIndex, int numVisibleNeurons, int numHiddenNeurons, bool convertActivationsToStates,
            /*TElementType epsilonw = (TElementType) 0.001, TElementType epsilonvb = (TElementType) 0.001,
            TElementType epsilonhb = (TElementType) 0.001,*/ TElementType weightcost = (TElementType) 0.0002,
            TElementType initialMomentum = (TElementType) 0.5, TElementType finalMomentum = (TElementType) 0.9, TElementType encodingNoiseLevel = (TElementType)1, TElementType decodingNoiseLevel = (TElementType)1)
            : base(
                gpu, rand, layerIndex, numVisibleNeurons, numHiddenNeurons, /*epsilonw, epsilonvb, epsilonhb,*/ weightcost,
                initialMomentum, finalMomentum)
        {
            ConvertActivationsToStates = convertActivationsToStates;
            _decodingNoiseLevel = decodingNoiseLevel;
            _encodingNoiseLevel = encodingNoiseLevel;
        }

        public override Matrix2D<TElementType> Encode(Matrix2D<TElementType> data)
        {
            int numcases = data.GetLength(0);
            using (Matrix2D<TElementType> tiledHiddenBiases = AsCuda.HiddenBiases.RepMatRows(numcases))
            {
                using (Matrix2D<TElementType> datavishid = data.Multiply(AsCuda.Weights))
                using (Matrix2D<TElementType> poshidprobs = datavishid.Subtract(tiledHiddenBiases))
                {
                    poshidprobs.LogisticInPlace();
                    using (
                        Matrix2D<TElementType> rand = AsCuda.GPU.UniformDistribution(AsCuda.GPURAND, numcases,
                            NumHiddenNeurons, (TElementType)_encodingNoiseLevel))
                    {
                        //end positive phase
                        return poshidprobs.GreaterThan(rand);
                    }
                }
            }
        }

        public override Matrix2D<TElementType> Decode(Matrix2D<TElementType> activations)
        {
            int numcases = activations.GetLength(0);
            using (Matrix2D<TElementType> tiledVisibleBiases = AsCuda.VisibleBiases.RepMatRows(numcases))
            using (Matrix2D<TElementType> weightsTransposed = AsCuda.Weights.Transpose())
            using (
                Matrix2D<TElementType> poshidstatesweightstransposed =
                    activations.Multiply(weightsTransposed))
            {
                Matrix2D<TElementType> negdata = poshidstatesweightstransposed.Subtract(tiledVisibleBiases);
                negdata.LogisticInPlace();

                if (ConvertActivationsToStates)
                {
                    using (negdata)
                    using (var rnd = AsCuda.GPU.UniformDistribution(AsCuda.GPURAND, numcases, NumVisibleNeurons, (TElementType)_decodingNoiseLevel))
                    {
                        negdata = negdata.GreaterThan(rnd);
                    }
                }

                return negdata;

            }
        }


        public override void GreedyTrain(Matrix2D<TElementType> data,
            IExitConditionEvaluator<TElementType> exitConditionEvaluator, ILearningRateCalculator<TElementType> weightLearningRateCalculator, ILearningRateCalculator<TElementType> hidBiasLearningRateCalculator, ILearningRateCalculator<TElementType> visBiasLearningRateCalculator)
        {
            exitConditionEvaluator.Start();
            var sw = new Stopwatch();
            int numcases = data.GetLength(0);
            TElementType error;
            int epoch;
            using (Matrix2D<TElementType> dataTransposed = data.Transpose())
            using (Matrix2D<TElementType> posvisact = data.SumColumns())
            {
                for (epoch = 0; ; epoch++)
                {
                    sw.Restart();
                    //start positive phase
                    using (Matrix2D<TElementType> tiledHiddenBiases = AsCuda.HiddenBiases.RepMatRows(numcases))
                    {
                        Matrix2D<TElementType> poshidstates, poshidact, posprods;
                        using (Matrix2D<TElementType> datavishid = data.Multiply(AsCuda.Weights))
                        using (Matrix2D<TElementType> poshidprobs = datavishid.Subtract(tiledHiddenBiases))
                        {
                            poshidprobs.LogisticInPlace();
                            poshidact = poshidprobs.SumColumns();
                            posprods = dataTransposed.Multiply(poshidprobs);
                            using (
                                Matrix2D<TElementType> rand = AsCuda.GPU.UniformDistribution(AsCuda.GPURAND, numcases,
                                    NumHiddenNeurons, (TElementType)_encodingNoiseLevel))
                            {
                                //end positive phase
                                poshidstates = poshidprobs.GreaterThan(rand);
                            }


                        }

                        //start negative phase
                        Matrix2D<TElementType> negdata, negprods, neghidact, negvisact;
                        using (Matrix2D<TElementType> tiledVisibleBiases = AsCuda.VisibleBiases.RepMatRows(numcases))
                        using (Matrix2D<TElementType> weightsTransposed = AsCuda.Weights.Transpose())
                        using (
                            Matrix2D<TElementType> poshidstatesweightstransposed =
                                poshidstates.Multiply(weightsTransposed))
                        {
                            poshidstates.Dispose();

                            negdata = poshidstatesweightstransposed.Subtract(tiledVisibleBiases);
                            negdata.LogisticInPlace();

                            using (Matrix2D<TElementType> negdataWeights = negdata.Multiply(AsCuda.Weights))
                            using (Matrix2D<TElementType> neghiddenprobs = negdataWeights.Subtract(tiledHiddenBiases))
                            {
                                neghiddenprobs.LogisticInPlace();
                                using (var negdataTransposed = negdata.Transpose())
                                {
                                    negprods = negdataTransposed.Multiply(neghiddenprobs);
                                    neghidact = neghiddenprobs.SumColumns();
                                    negvisact = negdata.SumColumns();
                                }
                            }

                        }

                        //end negative phase

                        using (Matrix2D<TElementType> delta = data.Subtract(negdata))
                        {
                            negdata.Dispose();

                            delta.PowInPlace((TElementType)2);
                            using (var errCols = delta.SumColumns())
                            using (var errrows = errCols.SumRows())
                                error = errrows.CopyLocal()[0, 0];
                        }

                        TElementType momentum = epoch > 5 ? FinalMomentum : InitialMomentum;

                        using (Matrix2D<TElementType> momentumvishidinc = _vishidinc.Multiply(momentum))
                        using (Matrix2D<TElementType> posprodsminusnegprods = posprods.Subtract(negprods))
                        using (Matrix2D<TElementType> weightcostWeight = AsCuda.Weights.Multiply(WeightCost))
                        {
                            posprods.Dispose();
                            negprods.Dispose();
                            posprodsminusnegprods.MultiplyInPlace((TElementType)1 / (TElementType)numcases);
                            posprodsminusnegprods.SubtractInPlace(weightcostWeight);
                            posprodsminusnegprods.MultiplyInPlace(weightLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch));
                            _vishidinc.Dispose();
                            _vishidinc = momentumvishidinc.Add(posprodsminusnegprods);
                        }


                        using (Matrix2D<TElementType> momentumvisbiasinc = _visbiasinc.Multiply(momentum))
                        using (Matrix2D<TElementType> posvisactminusnegvisact = posvisact.Subtract(negvisact))
                        {
                            negvisact.Dispose();
                            posvisactminusnegvisact.MultiplyInPlace(visBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch) / numcases);
                            _visbiasinc.Dispose();
                            _visbiasinc = momentumvisbiasinc.Add(posvisactminusnegvisact); ;
                        }

                        using (Matrix2D<TElementType> momentumhidbiasinc = _hidbiasinc.Multiply(momentum))
                        using (Matrix2D<TElementType> poshidactminusneghidact = poshidact.Subtract(neghidact))
                        {
                            neghidact.Dispose();
                            poshidactminusneghidact.MultiplyInPlace(hidBiasLearningRateCalculator.CalculateLearningRate(LayerIndex, epoch) / numcases);
                            _hidbiasinc.Dispose();
                            _hidbiasinc = momentumhidbiasinc.Add(poshidactminusneghidact);
                        }

                        AsCuda.Weights.AddInPlace(_vishidinc);
                        AsCuda.VisibleBiases.AddInPlace(_visbiasinc);
                        AsCuda.HiddenBiases.AddInPlace(_hidbiasinc);
                        poshidstates.Dispose();

                    }

                    OnEpochComplete(new EpochEventArgs<TElementType>()
                    {
                        Epoch = epoch,
                        Error = error,
                        Layer = LayerIndex
                    });

                    if (exitConditionEvaluator.Exit(epoch, error, sw.Elapsed))
                        break;
                }

                OnTrainComplete(new EpochEventArgs<TElementType>()
                {
                    Epoch = epoch,
                    Error = error,
                    Layer = LayerIndex
                });
            }

        }
    }
}