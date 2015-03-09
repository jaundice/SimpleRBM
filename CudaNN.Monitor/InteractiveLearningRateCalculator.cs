using SimpleRBM.Common;

namespace CudaNN.DeepBelief
{
    public class InteractiveLearningRateCalculator<T> : ILearningRateCalculator<T>
    {
        public T LearningRate { get; set; }

        public T CalculateLearningRate(int layer, int epoch)
        {
            return LearningRate;
        }
    }
}