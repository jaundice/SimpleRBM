using System;
using System.Linq;
using System.Text;

namespace CudaRbm
{
    public static class ExtensionClasses
    {
        public static void PrintMap(this float[,] arr)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));
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

        public static void PrintKaggleMap(this float[,] arr, int[] labels = null, float[,] reference = null)
        {
            var dataWidth = (int)Math.Sqrt(arr.GetLength(1));



            for (int i = 0; i < arr.GetLength(0); i++)
            {
                Console.WriteLine();
                if (labels != null)
                    Console.WriteLine(labels[i]);

                var builder1 = new StringBuilder();
                var builder2 = new StringBuilder();

                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    //if (j % dataWidth == 0)
                    //{
                    //    builder1.AppendLine();
                    //    if (reference != null)
                    //        builder2.AppendLine();
                    //}
                    //var num = arr[i, j];

                    builder1.Append(GetCharFor(arr[i, j]));
                    if (reference != null)
                        builder2.Append(GetCharFor(reference[i, j]));

                }

                var l1 = builder1.ToString();
                var l2 = builder2.ToString();

                for (var line = 0; line < dataWidth; line++)
                {
                    var line1 = new string(l1.Skip(line * dataWidth).Take(dataWidth).ToArray());
                    var line2 = reference == null ? null : new string(l2.Skip(line * dataWidth).Take(dataWidth).ToArray());

                    if (reference == null)
                    {
                        Console.WriteLine(line1);
                    }
                    else
                    {
                        Console.WriteLine("{0}       {1}", line1, line2);
                    }
                }

                Console.WriteLine();
            }
        }

        private static string GetCharFor(float f)
        {
            return f == 1f ? "1" : f > 0f ? "." : "0";
        }


    }
}