using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CudaNN;

namespace SimpleRBM.Test
{
    [TestClass]
    public class DeserializeTest : CudaTestBase
    {
        [TestMethod]
        public void TestMethod1()
        {
            var path = Path.Combine(Environment.CurrentDirectory, "deserialzationtest1.dat");

            var rbm = new CudaAdvancedRbmBinary(_dev, _rand, 1, 50, 100, true);

            Console.WriteLine("Original Type is {0}", rbm.GetType());

            rbm.Save(path);

            rbm = null;
            var obj = (CudaAdvancedRbmBase)CudaAdvancedRbmBase.Deserialize(path, _dev, _rand);
            Console.WriteLine("Deserialized Type is {0}", obj.GetType().FullName);
        }
    }
}