using System.IO;
using MultidimRBM;
using SimpleRBM.Common;

namespace SimpleRBM.MultiDim
{
    public class MultiDimDbnFactory : IDeepBeliefNetworkFactory<double>
    {
        public IDeepBeliefNetwork<double> Create(DirectoryInfo networkDataDir, int[] appendLayers = null,
            double learningRate = 0.2,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory = null)
        {
            return new DeepBeliefNetworkD(networkDataDir, learningRate, exitConditionEvaluatorFactory, appendLayers);
        }

        public IDeepBeliefNetwork<double> Create(int[] layerSizes = null, double learningRate = 0.2,
            IExitConditionEvaluatorFactory<double> exitConditionEvaluatorFactory = null)
        {
            return new DeepBeliefNetworkD(layerSizes, learningRate, exitConditionEvaluatorFactory);
        }
    }
}