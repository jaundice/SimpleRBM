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

            //var factory = new CudaDbnFactory();
            var factory = new MultiDimDbnFactory();

            if (Environment.GetCommandLineArgs().Contains("-mnist"))
            {
                var demo = new MNist();
                demo.Execute(new CudaDbnFactory(), new ManualKeyPressExitEvaluatorFactory<float>(0.0005f, 200000),
                    new[] { 250 * 250, 2000, 2000 }, new MNistData(), learningRate, trainingSize, skipTrainingRecords);
            }
            else if (Environment.GetCommandLineArgs().Contains("-kaggle"))
            {
                var demo = new Kaggle();
                demo.Execute<float, int>(new CudaDbnFactory(),
                    new ManualKeyPressExitEvaluatorFactory<float>(0.0005f, 2000000),
                    new[] { 784, 1568, 784, 64, 10 }, new KaggleData(), learningRate, trainingSize, skipTrainingRecords);
            }
            else
            {
                var demo = new HandwrittenNumbers();
                demo.Execute<double, int>(new MultiDimDbnFactory(),
                    new ManualKeyPressExitEvaluatorFactory<double>(0.0005f, 2000000),
                    new[] { 1024, 1024, 1024, 10 }, new HandwrittenNumbersData(), learningRate, trainingSize,
                    skipTrainingRecords);
            }
        }
    }
}