using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleRBM.Demo.Demo;

namespace SimpleRBM.Test
{
    [TestClass]
    public class ImageReaderTest
    {
        [TestMethod]
        public void TestImages()
        {
            var provider = new MNistData();
            string[] labels;
            var data = provider.ReadTrainingData(@"E:\Dev\GitHub\lfw-deepfunneled\lfw-deepfunneled", 0, 10, out labels);

            provider.PrintToScreen(data, labels);
        }
    }
}
