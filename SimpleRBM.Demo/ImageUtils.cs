using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Threading.Tasks;

namespace SimpleRBM.Demo
{
    public static class ImageUtils
    {
        public delegate T ConvertPixel<T>(IntPtr startAddress, int stride, int x, int y);

        public static unsafe float ConvertRGBToGreyF(IntPtr startAddress, int stride, int x, int y)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * 3);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return ((R * 0.3f) + (G * 0.59f) + (B * 0.11f)) / 255f;
        }

        public static unsafe double ConvertRGBToGreyD(IntPtr startAddress, int stride, int x, int y)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * 3);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return ((R * 0.3) + (G * 0.59) + (B * 0.11)) / 255.0;
        }

        public static IEnumerable<T[]> ReadImageData<T>(IEnumerable<FileInfo> files,
            ConvertPixel<T> pixelConverter)
        {
            return files.Select(a => ReadImageFile(a, pixelConverter));
        }

        private static T[] ReadImageFile<T>(FileInfo fileInfo, ConvertPixel<T> pixelConverter)
        {
            using (var img = (Bitmap)Image.FromFile(fileInfo.FullName))
            {
                BitmapData data = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadOnly,
                    img.PixelFormat);


                try
                {
                    var bytes = new T[img.Width * img.Height];

                    var w = img.Width;
                    var h = img.Height;


                    Parallel.For(0, w, a => Parallel.For(0, h, b =>
                    {
                        bytes[b * w + a] = pixelConverter(data.Scan0, data.Stride, a, b);
                    }));

                    return bytes;
                }
                finally
                {
                    img.UnlockBits(data);
                }
            }
        }

        public static unsafe void SaveImageData<T>(T[,] data, int sourceRow, string path, Func<T, byte> pixelConverter)
        {
            var dimension = (int)Math.Ceiling(Math.Sqrt(data.GetLength(1)));
            var rowLength = data.GetLength(1);
            using (var bmp = new Bitmap(dimension, dimension))
            {
                for (var i = 0; i < dimension; i++)
                {
                    for (var j = 0; j < dimension; j++)
                    {
                        var idx = j * dimension + i;
                        var intensity = idx < rowLength ? pixelConverter(data[sourceRow, idx]) : 0;
                        bmp.SetPixel(i, j, Color.FromArgb(intensity, intensity, intensity));
                    }
                }

                bmp.Save(path, ImageFormat.Jpeg);
            }
        }
    }
}