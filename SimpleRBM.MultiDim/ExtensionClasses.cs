using System;
using System.Linq;

namespace MultidimRBM
{
    public static class ExtensionClasses
    {
        public static void PrintMap(this double[,] arr)
        {
            int dataWidth = (int)Math.Sqrt(arr.GetLength(1));
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    if (j % dataWidth == 0)
                        Console.WriteLine();
                    Console.Write(arr[i, j].ToString("N0"));
                }
                Console.WriteLine();

            }
        }

        //public static void PrintMap(this double[,] mat, int rows)
        //{
        //    mat.ToArray().Flatten().PrintMap(rows);
        //}

        //public static double[] Flatten(this double[][] arr)
        //{
        //    return arr.SelectMany(x => x).ToArray();
        //}
    }
}
