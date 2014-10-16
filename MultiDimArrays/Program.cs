using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultidimRBM
{
    internal class Program
    {
        private static void Main()
        {
            //Our dataset cosists of images of handwritten digits (0-9)
            //Let's only take 100 of those for training
            double[][] trainingData = DataParser.Parse("optdigits-tra.txt").ToArray();

            //Although it is tempting to say that the final hidden layer has 10 features (10 numbers) but let's keep it real.
            var rbm = new DeepBeliefNetworkD(new[] { 1024, 512, 10 }, 0.1);
            //var rbm = new DeepBeliefNetworkD(new[] { 1024, 128, 16 }, 0.3);


            rbm.TrainAll(Matrix2D.JaggedToMultidimesional(trainingData.Take(200).ToArray())/*, 1500, 5*/);



            Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
            Console.WriteLine("Reconstructions");
            Console.WriteLine();
            //Take a sample of input arrays and try to reconstruct them.
            double[,] reconstructedItems = rbm.Reconstruct(Matrix2D.JaggedToMultidimesional(trainingData.Skip(200).Take(50).ToArray()));

            reconstructedItems.PrintMap();

            //reconstructedItems.ToList().ForEach(x =>
            //{
            //    Console.WriteLine("");
            //    x.PrintMap();
            //});
            Console.WriteLine();
            Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
            Console.WriteLine("Daydream");

            do
            {
                //Day dream 10 images
                rbm.DayDream(10).PrintMap();

                Console.WriteLine();
                Console.WriteLine("++++++++++++++++++++++++++++++++++++++++++++");
            } while (!new[] { 'Q', 'q' }.Contains(Console.ReadKey().KeyChar));
        }
    }

    public static class DataParser
    {

        public static double[][] Parse(string filePath)
        {
            var x = File.ReadAllText(filePath);

            x = x.Replace("\r\n", "");

            var y = x.Split(" ".ToCharArray());

            var t =
                y.Select(
                    s =>
                    s.Substring(1).PadRight(1024, '0').Select(
                        n => double.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();

            return t;
        }


    }
}
