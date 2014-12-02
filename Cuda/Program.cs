using System;
using System.Linq;
using CudaRbm.Demo;

namespace CudaRbm
{
    internal class Program
    {
        private static void Main()
        {
            if (Environment.GetCommandLineArgs().Contains("-mnist"))
            {
                Mnist.Execute();
            }
            else if (Environment.GetCommandLineArgs().Contains("-kaggle"))
            {
                Kaggle.Execute();
            }
            else
            {
                HandwrittenNumbers.Execute();
            }
        }
    }
}