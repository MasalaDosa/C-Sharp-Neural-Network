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
        public void TestTranspose1()
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


        [Fact]
        public void TestTranspose2()
        {
            Matrix x = new Matrix(new double[4, 6] {
                {1, 2, 3, 4, 5, 6},
                {7, 8 ,9, 10, 11, 12},
                {13, 14 ,15, 16, 17, 18},
                {19, 20, 21, 22, 23, 24}
            });

            Matrix z = x.Transpose();

            AssertIsAsExpected(z, new double[6, 4] {
                {1, 7, 13, 19},
                {2, 8, 14, 20},
                {3, 9, 15, 21},
                {4, 10, 16, 22},
                {5, 11, 17, 23},
                {6, 12, 18, 24}
            });
        }


        [Fact]
        public void TestTranspose3()
        {
            Matrix x = new Matrix(new double[1, 6] {
                {1, 2, 3, 4, 5, 6}
            });

            Matrix z = x.Transpose();

            AssertIsAsExpected(z, new double[6, 1] {
                {1},
                {2},
                {3},
                {4},
                {5},
                {6}
            });
        }


        [Fact]
        public void TestTranspose4()
        {
            Matrix x = new Matrix(new double[6, 1] {
                {1},
                {2},
                {3},
                {4},
                {5},
                {6}
            });

            Matrix z = x.Transpose();

            AssertIsAsExpected(z, new double[1, 6] {
                {1, 2, 3, 4, 5, 6}
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
                    Assert.True(z.Data[i, j] - expected[i, j] < 0.001, $"{z.ToString()}");
                }
            }
        }
    }
}
