using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Mono.CSharp;

namespace SimpleRBM.Demo
{
    public static class ImageUtils
    {
        public delegate T ConvertPixel<T>(IntPtr startAddress, int stride, int x, int y, int pixelWidth);

        public static unsafe float ConvertRGBToGreyIntF(IntPtr startAddress, int stride, int x, int y, int pixelWidth)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * pixelWidth);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return ((R * 0.3f) + (G * 0.59f) + (B * 0.11f));
        }

        public static unsafe float ConvertRGBToGreyF(IntPtr startAddress, int stride, int x, int y, int pixelWidth)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * pixelWidth);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return ((R * 0.3f) + (G * 0.59f) + (B * 0.11f)) / 255f;
        }

        public static unsafe float ConvertRGBToGreyPosNegF(IntPtr startAddress, int stride, int x, int y, int pixelWidth)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * pixelWidth);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return -0.5f + ((R * 0.3f) + (G * 0.59f) + (B * 0.11f)) / 255f;
        }

        //public static unsafe double ConvertRGBToGreyD(IntPtr startAddress, int stride, int x, int y, int pixelWidth)
        //{
        //    var data = (byte*)startAddress;
        //    int ost = y * stride + (x * pixelWidth);

        //    byte B = data[ost];
        //    byte G = data[ost + 1];
        //    byte R = data[ost + 2];

        //    return ((R * 0.3) + (G * 0.59) + (B * 0.11)) / 255.0;
        //}

        public static unsafe double ConvertRGBToGreyD(IntPtr startAddress, int stride, int x, int y, int pixelWidth)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * pixelWidth);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return ((R * 0.3) + (G * 0.59) + (B * 0.11)) / 255.0;
        }



        public static unsafe double ConvertRGBToGreyPosNegD(IntPtr startAddress, int stride, int x, int y,
            int pixelWidth)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * pixelWidth);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            return -0.5 + ((R * 0.3) + (G * 0.59) + (B * 0.11)) / 255.0;
        }

        public static unsafe double ConvertRGBToGreyIntD(IntPtr startAddress, int stride, int x, int y, int pixelWidth)
        {
            var data = (byte*)startAddress;
            int ost = y * stride + (x * pixelWidth);

            byte B = data[ost];
            byte G = data[ost + 1];
            byte R = data[ost + 2];

            var r = (double)(byte)(((double)R * 0.3) + ((double)G * 0.59) + ((double)B * 0.11));

            return r;
        }

        public static IEnumerable<T[]> ReadImageData<T>(IEnumerable<FileInfo> files,
            ConvertPixel<T> pixelConverter)
        {
            return files.Select(a => ReadImageFile(a, pixelConverter));
        }

        public static void CopyImageDataTo<T>(string filePath, T[,] target, int targetRow, int rowOffset, ConvertPixel<T> pixelConverter, Func<T, T> sourceToTargetConverter)
        {
            using (var img = (Bitmap)Image.FromFile(filePath))
            {
                BitmapData data = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadOnly,
                    img.PixelFormat);

                int pixWidth = 0;
                switch (img.PixelFormat)
                {
                    case PixelFormat.Format24bppRgb:
                        {
                            pixWidth = 3;
                            break;
                        }
                    case PixelFormat.Format32bppRgb:
                        {
                            pixWidth = 4;
                            break;
                        }
                    default:
                        throw new NotImplementedException();
                }
                try
                {
                    var w = img.Width;
                    var h = img.Height;


                    Parallel.For(0, w,
                        a =>
                            Parallel.For(0, h,
                                b => { target[targetRow, b * w + a + rowOffset] = sourceToTargetConverter(pixelConverter(data.Scan0, data.Stride, a, b, pixWidth)); }));
                }
                finally
                {
                    img.UnlockBits(data);
                }

            }
        }

        private static T[] ReadImageFile<T>(FileInfo fileInfo, ConvertPixel<T> pixelConverter)
        {
            using (var img = (Bitmap)Image.FromFile(fileInfo.FullName))
            {
                BitmapData data = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadOnly,
                    img.PixelFormat);

                int pixWidth = 0;
                switch (img.PixelFormat)
                {
                    case PixelFormat.Format24bppRgb:
                        {
                            pixWidth = 3;
                            break;
                        }
                    case PixelFormat.Format32bppRgb:
                        {
                            pixWidth = 4;
                            break;
                        }
                    default:
                        throw new NotImplementedException();
                }
                T[] bytes;
                try
                {
                    bytes = new T[img.Width * img.Height];

                    var w = img.Width;
                    var h = img.Height;


                    Parallel.For(0, w,
                        a =>
                            Parallel.For(0, h,
                                b => { bytes[b * w + a] = pixelConverter(data.Scan0, data.Stride, a, b, pixWidth); }));
                }
                finally
                {
                    img.UnlockBits(data);
                }
                return bytes;
            }
        }

        public static unsafe void SaveImageData<T>(T[,] data, int sourceRow, string path, Func<T, byte> pixelConverter)
        {
            int stride;
            using (var bmp = GenerateBitmap(data, sourceRow, pixelConverter, out stride))
                bmp.Save(path, ImageFormat.Jpeg);
        }

        public static unsafe Bitmap GenerateBitmap<T>(T[,] data, int sourceRow, Func<T, byte> pixelConverter,
            out int stride)
        {
            var dimension = (int)Math.Ceiling(Math.Sqrt(data.GetLength(1)));
            var rowLength = data.GetLength(1);

            var bmp = new Bitmap(dimension, dimension, PixelFormat.Format24bppRgb);
            var w = bmp.Width;
            var h = bmp.Height;
            BitmapData imgData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.WriteOnly,
                bmp.PixelFormat);
            stride = imgData.Stride;
            var pixelSize = 3;
            try
            {
                byte* row = (byte*)imgData.Scan0;
                //Parallel.For(0, w, ww =>
                //Parallel.For(0, h, hh =>
                for (var ww = 0; ww < w; ww++)
                    for (var hh = 0; hh < h; hh++)
                    {
                        var idx = hh * dimension + ww;
                        var intensity = idx < rowLength ? pixelConverter(data[sourceRow, idx]) : 0;
                        row[hh * imgData.Stride + ww * pixelSize] = (byte)intensity;
                        row[hh * imgData.Stride + ww * pixelSize + 1] = (byte)intensity;
                        row[hh * imgData.Stride + ww * pixelSize + 2] = (byte)intensity;
                        //}
                    }
                //))
                ;
            }
            finally
            {
                bmp.UnlockBits(imgData);
            }

            return bmp;
        }
    }
}