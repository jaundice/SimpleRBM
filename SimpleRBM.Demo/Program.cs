using System;
using System.Linq;
using SimpleRBM.Common.ExitCondition;
using SimpleRBM.Cuda;
using SimpleRBM.Demo.Demo;
using SimpleRBM.MultiDim;

namespace SimpleRBM.Demo
{
    internal class Program
    {
        private static void Main()
        {
            float learningRate = CommandLine.ReadCommandLine("-learningrate:", float.TryParse, 0.2f);
            int trainingSize = CommandLine.ReadCommandLine("-trainingsize:", int.TryParse, 2048);
            int skipTrainingRecords = CommandLine.ReadCommandLine("-skiptrainingrecords:", int.TryParse, 0);

            if (Environment.GetCommandLineArgs().Contains("-mnist"))
            {
                Mnist.Execute();
            }
            else if (Environment.GetCommandLineArgs().Contains("-kaggle"))
            {
                var demo = new Kaggle();
                demo.Execute(new MultiDimDbnFactory(),
                    new ManualKeyPressExitEvaluatorFactory<double>(0.00005f, 2000000),
                    new[] {784, 500, 500, 2000, 10}, new KaggleData(), learningRate, trainingSize, skipTrainingRecords);
            }
            else
            {
                var demo = new HandwrittenNumbers();
                demo.Execute(new MultiDimDbnFactory(), new ManualKeyPressExitEvaluatorFactory<double>(0.000002f, 2000000),
                    new[] {1024, 1024, 1024, 10}, new HandwrittenNumbersData(), learningRate, trainingSize,
                    skipTrainingRecords);
            }
        }
    }
}