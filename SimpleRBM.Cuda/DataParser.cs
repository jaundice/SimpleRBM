using System.Globalization;
using System.IO;
using System.Linq;

namespace SimpleRBM.Common
{
    //public static class DataParser
    //{
    //    public static float[][] Parse(string filePath)
    //    {
    //        string x = File.ReadAllText(filePath);

    //        x = x.Replace("\r\n", "");

    //        string[] y = x.Split(" ".ToCharArray());

    //        float[][] t =
    //            y.Select(
    //                s =>
    //                    s.Substring(1).PadRight(1024, '0').Select(
    //                        n => float.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();

    //        return t;
    //    }

    //    public static double[][] Parse(string filePath)
    //    {
    //        string x = File.ReadAllText(filePath);

    //        x = x.Replace("\r\n", "");

    //        string[] y = x.Split(" ".ToCharArray());

    //        float[][] t =
    //            y.Select(
    //                s =>
    //                    s.Substring(1).PadRight(1024, '0').Select(
    //                        n => float.Parse(n.ToString(CultureInfo.InvariantCulture))).ToArray()).ToArray();

    //        return t;
    //    }
    //}
}