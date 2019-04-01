using System;
using Xunit;

namespace MatrixLib.Test
{
    public class TestMatrix
    {
        [Fact]
        public void TestDotProduct2by2dot2by2()
        {
            Matrix x = new Matrix(new double[2, 2] {
                {1, 2 },
                {3, 4 }
            });

            Matrix y = new Matrix(new double[2, 2] {
                {5, 6 },
                {7, 8 }
            });

            Matrix z = x.DotProduct(y);

            double[,] expected = {
                { 19, 22 },
                { 43, 50}
            };
            AssertIsAsExpected(z, expected);

        }


        [Fact]
        public void TestDotProduct3by3dot3by1()
        {
            Matrix x = new Matrix(new double[3, 3] {
                { 0.9, 0.3, 0.4 },
                { 0.2, 0.8, 0.2 },
                { 0.1, 0.5, 0.6 }
            });

            Matrix y = new Matrix(new double[3, 1] {
                { 0.9 },
                { 0.1 },
                { 0.8 }
            });



            Matrix z = x.DotProduct(y);

            double[,] expected = {
                { 1.16 },
                { 0.42 },
                { 0.62 }
            };
            AssertIsAsExpected(z, expected);

        }


        [Fact]
        public void TestDotProduct3by3dot3by2()
        {
            Matrix x = new Matrix(new double[3, 3] {
                { 0.9, 0.3, 0.4 },
                { 0.2, 0.8, 0.2 },
                { 0.1, 0.5, 0.6 }
            });

            Matrix y = new Matrix(new double[3, 2] {
                { 0.9, 0.1 },
                { 0.1, 0.2 },
                { 0.8, 0.3 }
            });



            Matrix z = x.DotProduct(y);

            double[,] expected = {
                { 1.16, 0.27 },
                { 0.42, 0.24 },
                { 0.62, 0.29 }
            };
            AssertIsAsExpected(z, expected);

        }


        [Fact]
        public void TestDotProduct3by2dot3by3()
        {

            Matrix x = new Matrix(new double[3, 2] {
                { 0.9, 0.1 },
                { 0.1, 0.2 },
                { 0.8, 0.3 }
            });

            Matrix y = new Matrix(new double[3, 3] {
                { 0.9, 0.3, 0.4 },
                { 0.2, 0.8, 0.2 },
                { 0.1, 0.5, 0.6 }
            });

            Assert.Throws<ArgumentException>(() => x.DotProduct(y));
        }


        [Fact]
        public void TestAddition2by2plus2by2()
        {
            Matrix x = new Matrix(new double[2, 2] {
                {1, 2},
                {3, 4}
            });

            Matrix y = new Matrix(new double[2, 2] {
                {5, 6},
                {7, 8}
            });


            Matrix z = x.Add(y);

            AssertIsAsExpected(z, new double[2, 2] {
                {6, 8},
                {10, 12}
            });
        }


        [Fact]
        public void TestAddition3by2plus3by3()
        {

            Matrix x = new Matrix(new double[3, 2] {
                { 0.9, 0.1 },
                { 0.1, 0.2 },
                { 0.8, 0.3 }
            });

            Matrix y = new Matrix(new double[3, 3] {
                { 0.9, 0.3, 0.4 },
                { 0.2, 0.8, 0.2 },
                { 0.1, 0.5, 0.6 }
            });

            Assert.Throws<ArgumentException>(() => x.Add(y));
        }


        [Fact]
        public void TestSubtract2by2minus2by2()
        {
            Matrix x = new Matrix(new double[2, 2] {
                {1, 2},
                {3, 4}
            });

            Matrix y = new Matrix(new double[2, 2] {
                {5, 6},
                {7, 8}
            });


            Matrix z = x.Subtract(y);

            AssertIsAsExpected(z, new double[2, 2] {
                {-4, -4},
                {-4, -4}
            });
        }


        [Fact]
        public void TestSubtract3by2minus3by3()
        {

            Matrix x = new Matrix(new double[3, 2] {
                { 0.9, 0.1 },
                { 0.1, 0.2 },
                { 0.8, 0.3 }
            });

            Matrix y = new Matrix(new double[3, 3] {
                { 0.9, 0.3, 0.4 },
                { 0.2, 0.8, 0.2 },
                { 0.1, 0.5, 0.6 }
            });

            Assert.Throws<ArgumentException>(() => x.Subtract(y));
        }


        [Fact]
        public void TestCellwiseMultiply2by2times2by2()
        {
            Matrix x = new Matrix(new double[2, 2] {
                {1, 2},
                {3, 4}
            });

            Matrix y = new Matrix(new double[2, 2] {
                {5, 6},
                {7, 8}
            });


            Matrix z = x.CellByCellProduct(y);

            AssertIsAsExpected(z, new double[2, 2] {
                {5, 12},
                {21, 32}
            });
        }


        [Fact]
        public void TestCellwiseMultiply3by2times3by3()
        {

            Matrix x = new Matrix(new double[3, 2] {
                { 0.9, 0.1 },
                { 0.1, 0.2 },
                { 0.8, 0.3 }
            });

            Matrix y = new Matrix(new double[3, 3] {
                { 0.9, 0.3, 0.4 },
                { 0.2, 0.8, 0.2 },
                { 0.1, 0.5, 0.6 }
            });

            Assert.Throws<ArgumentException>(() => x.CellByCellProduct(y));
        }


        [Fact]
        public void TestApplyFunction3by2times5()
        {

            Matrix x = new Matrix(new double[3, 2] {
                { 0.9, 0.1 },
                { 0.1, 0.2 },
                { 0.8, 0.3 }
            });

            Matrix z = x.ApplyFunction(d => d * 5);

            AssertIsAsExpected(z, new double[3, 2] {
                {4.5, 0.5},
                {0.5, 1.0},
                {4.8, 1.5}
            });
        }


        [Fact]
        public void Transpose()
        {
            Matrix x = new Matrix(new double[2, 3] {
                {1, 2, 3},
                {4, 5 ,6}
            });

            Matrix z = x.Transpose();

            AssertIsAsExpected(z, new double[3, 2] {
                {1, 4},
                {2, 5},
                {3, 6}
            });
        }


        static void AssertIsAsExpected(Matrix z, double[,] expected)
        {
            Assert.True(z.NumberOfRows == expected.GetLength(0));
            Assert.True(z.NumberOfColumns == expected.GetLength(1));
            for (int i = 0; i < z.NumberOfRows; i++)
            {
                for (int j = 0; j < z.NumberOfColumns; j++)
                { 
                    Assert.True(z.Data[i, j] - expected[i, j] < 0.001);
                }
            }
        }
    }
}
