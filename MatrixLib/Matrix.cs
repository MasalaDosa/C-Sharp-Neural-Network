using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MatrixLib
{
    /// <summary>
    /// A simple, inefficient, matrix class with only the functionality needed to support a simple neural network.
    /// </summary>
    public class Matrix
    {
        public Matrix(double[,] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data), $"{nameof(data)} is required.");
            if (data.GetLength(0) < 1 || data.GetLength(1) < 1) throw new ArgumentException(nameof(data), $"{nameof(data)} has zero rows or columns.");
            Data = data;
        }


        /// <summary>
        /// Get the underlying Data.
        /// </summary>
        public double[,] Data { get; private set; }


        /// <summary>
        /// Number of rows
        /// </summary>
        public int NumberOfRows
        {
            get
            {
                return Data.GetLength(0);
            }
        }


        /// <summary>
        /// Number of columns
        /// </summary>
        public int NumberOfColumns
        {
            get
            {
                return Data.GetLength(1);
            }
        }


        /// <summary>
        /// Peform a DotProduct inefficiently.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public Matrix DotProduct(Matrix other)
        {
            // The other matrix must exist and be compatible (it should have the same number of rows that we do columns)
            if (other == null) throw new ArgumentNullException(nameof(other));
            if (NumberOfColumns != other.NumberOfRows) throw new ArgumentException(nameof(other), $"{nameof(other)} is incompatible for dot product.");
           
            // The resulting matrix will have the same number of rows as we do, but the same number of columns that other has.
            double[,] resultData = new double[NumberOfRows, other.NumberOfColumns];

            // Calculate the resul cells.
            for (int thisRow = 0; thisRow < Data.GetLength(0); thisRow++)
            {
                for (int otherColumn = 0; otherColumn < other.Data.GetLength(1); otherColumn++)
                {
                    double total = 0.0d;
                    for (int thisColumn = 0; thisColumn < Data.GetLength(1); thisColumn++)
                    {

                        total += this.Data[thisRow, thisColumn] * other.Data[thisColumn, otherColumn];
                    }
                    resultData[thisRow, otherColumn] = total;
                }
            }
            return new Matrix(resultData);
        }


        public Matrix Subtract(Matrix other)
        {
            // The other matrix must exist and be compatible (it should have the same number of rows and columns)
            if (other == null) throw new ArgumentNullException(nameof(other));
            if (this.NumberOfColumns != other.NumberOfColumns || this.NumberOfRows != other.NumberOfRows) throw new ArgumentException(nameof(other), $"{nameof(other)} is incompatible for subtract.");

            double[,] resultData = new double[this.NumberOfRows, this.NumberOfColumns];
            for (int r = 0; r < this.NumberOfRows; r++)
            {
                for (int c = 0; c < this.NumberOfColumns; c++)
                {
                    resultData[r, c] = Data[r, c] - other.Data[r, c];
                }
            }
            return new Matrix(resultData);
        }


        public Matrix Add(Matrix other)
        {
            // The other matrix must exist and be compatible (it should have the same number of rows and columns)
            if (other == null) throw new ArgumentNullException(nameof(other));
            if (this.NumberOfColumns != other.NumberOfColumns || this.NumberOfRows != other.NumberOfRows) throw new ArgumentException(nameof(other), $"{nameof(other)} is incompatible for add.");

            double[,] resultData = new double[this.NumberOfRows, this.NumberOfColumns];
            for (int r = 0; r < this.NumberOfRows; r++)
            {
                for (int c = 0; c < this.NumberOfColumns; c++)
                {
                    resultData[r, c] = Data[r, c] + other.Data[r, c];
                }
            }
            return new Matrix(resultData);
        }


        public Matrix CellByCellProduct(Matrix other)
        {
            // The other matrix must exist and be compatible (it should have the same number of rows and columns)
            if (other == null) throw new ArgumentNullException(nameof(other));
            if (this.NumberOfColumns != other.NumberOfColumns || this.NumberOfRows != other.NumberOfRows) throw new ArgumentException(nameof(other), $"{nameof(other)} is incompatible for cell by cell product.");

            double[,] resultData = new double[this.NumberOfRows, this.NumberOfColumns];
            for (int r = 0; r < this.NumberOfRows; r++)
            {
                for (int c = 0; c < this.NumberOfColumns; c++)
                {
                    resultData[r, c] = Data[r, c] * other.Data[r, c];
                }
            }
            return new Matrix(resultData);
        }


        public Matrix Transpose()
        {
            var output = new double[this.NumberOfColumns, this.NumberOfRows];
            for (int i = 0; i < this.NumberOfRows; i++)
            {
                for (int j = 0; j < this.NumberOfColumns; j++)
                {
                    output[j, i] = this.Data[i, j];
                }
            }
            return new Matrix(output);
        }


        /// <summary>
        /// Apply a function to each element of the matrix.
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        public Matrix ApplyFunction(Func<double, double> f)
        {
            var output = new double[this.NumberOfRows, this.NumberOfColumns];
            for (int row = 0; row < this.NumberOfRows; row++)
            {
                for (int col = 0; col < this.NumberOfColumns; col++)
                {
                    output[row, col] = f(this.Data[row, col]);
                }
            }
            return new Matrix(output);
        }
        
        
        public Matrix Scale(double min, double max)
        {
            return ApplyFunction(d =>
            {
                d -= Min();
                d /= Max() - Min();
                d *= (max - min);
                d += min;
                return d;
            });
        }


        public double[] Row(int i)
        {
            if (i < 0 || i >= NumberOfRows) throw new ArgumentOutOfRangeException(nameof(i));
            double[] result = new double[NumberOfColumns];
            for (int c = 0; c < NumberOfColumns; c++)
            {
                result[c] = Data[i, c];
            }
            return result;
        }


        public double[] Col(int i)
        {
            return Transpose().Row(i);
        }


        public double Min()
        {
            return Data.Cast<double>().Min();
        }


        public double Max()
        {
            return Data.Cast<double>().Max();
        }


        public override string ToString()
        {
            StringBuilder result = new StringBuilder();
            result.AppendLine($"Matrix :{NumberOfRows} rows by {NumberOfColumns} columns");
            for (int r = 0; r < NumberOfRows; r++)
            {
                List<double> currentRow = new List<double>();
                for (int c = 0; c < NumberOfColumns; c++)
                {
                    currentRow.Add(Data[r, c]);
                }
                result.AppendLine(string.Join(",\t", currentRow));
            }
            return result.ToString();
        }


        public static Matrix OneRowNColFrom1DArray(double[] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data), $"{nameof(data)} is required.");
            if (data.GetLength(0) < 1) throw new ArgumentException(nameof(data), $"{nameof(data)} has zero elements.");
            // Build expected matrix - this will be N rows by 1 column
            double[,] matrixData = new double[1, data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                matrixData[0, i] = data[i];
            }
            return new Matrix(matrixData);
        }


        public static Matrix NRow1ColFrom1DArray(double[] data)
        {

            return OneRowNColFrom1DArray(data).Transpose();
        }
    }
}
