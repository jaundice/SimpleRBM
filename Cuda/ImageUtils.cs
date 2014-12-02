using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace CudaRbm
{
    public static class ImageUtils
    {
        public static IEnumerable<byte[]> ReadImageData(IEnumerable<FileInfo> files)
        {
            return files.Select(ReadImageFile);
        }

        private static byte[] ReadImageFile(FileInfo fileInfo)
        {
            using (var img = (Bitmap)Image.FromFile(fileInfo.FullName))
            {
                BitmapData data = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), ImageLockMode.ReadOnly,
                    img.PixelFormat);
                try
                {
                    var bytes = new byte[data.Stride * data.Height];
                    Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
                    return bytes;
                }
                finally
                {
                    img.UnlockBits(data);
                }
            }
        }

        private static void SaveImageData(byte[] data, string path)
        {
            using (var ms = new MemoryStream(data))
            using (Bitmap bmp = new Bitmap(ms))
            {
                bmp.Save(path);
            }
        }
    }
}