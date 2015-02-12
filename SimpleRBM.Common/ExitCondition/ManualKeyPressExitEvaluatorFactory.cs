using System;

namespace SimpleRBM.Common.ExitCondition
{
    public class ManualKeyPressExitEvaluatorFactory<T> : IExitConditionEvaluatorFactory<T> where T : struct, IComparable<T>
    {
        private readonly int _maxEpochs;
        private readonly T _minError;

        public ManualKeyPressExitEvaluatorFactory(double minError, int maxEpochs = 100000)
        {
            Console.WriteLine("Using Manual keypress exit evaluator");
            Console.WriteLine(@"RBM will exit training after:
{0} epochs or 
when error is lower than {1} or 
when a key press is detected.
If the pressed key is 'l' it will exit the next time the epoch error falls to its lowest seen, otherwise it will exit immediately", maxEpochs, minError);
            _maxEpochs = maxEpochs;
            _minError = (T) Convert.ChangeType(minError, typeof(T));
        }

        public IExitConditionEvaluator<T> Create(int layerIndex)
        {
            return new ManualKeyPressEvaluator<T>(layerIndex, _maxEpochs, _minError);
        }
    }
}