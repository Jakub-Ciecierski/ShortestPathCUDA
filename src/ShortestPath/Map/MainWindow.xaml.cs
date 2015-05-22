using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace MapGUI
{
    /// <summary>
    ///     Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        /// <summary>
        ///     Checks if mouse is being draged
        /// </summary>
        bool isLeftDrag = false;
        bool isRightDrag = false;

        bool isMiddleDrag = false;
        const double middleDragFactor = 0.5;
        System.Windows.Point middleDragStartPoint;

        int gridWidth = 150;
        int gridHeight = 150;

        Map map;

        private int speed;

        private const double zoomFactor = 0.1;
        private double zoomValue = 1.0;

        bool isMaximized = false;
        private Rect _restoreLocation;

        public MainWindow()
        {
            InitializeComponent();

            windowsSettings();

            map = new Map(automatonImage);
        }

        private void windowsSettings()
        {
            //this.ShowTitleBar = false;
        }


        private void Window_Loaded(object sender, RoutedEventArgs e)
        {

        }

        /*******************************************************************/
        /*************************** DROP FILE *****************************/
        /*******************************************************************/

        private void mapFromFile(string filename)
        {
            string[] lines = System.IO.File.ReadAllLines(filename, Encoding.GetEncoding("ISO-8859-1"));
            int l = 0;
            while (!lines[l++].Equals("MAP")) ;

            int i;

            string N = "";
            string M = "";

            bool turn = false;

            foreach (char c in lines[l]) 
            {
                if (Char.IsDigit(c))
                {
                    if (!turn)
                        N += c;
                    else
                        M += c;
                }
                if (c == ' ' && !N.Equals(""))
                    turn = true;
            }
            l++;
            int n = Int32.Parse(N);
            int m = Int32.Parse(M);

            int[][] map = new int[n][];
            for (i = 0; i < n; i++)
            {
                map[i] = new int[m];
            }

            string line;

            for (i = 0; i < n; i++)
            {
                int j = 0;
                line = lines[i+l];
                int digitIndex = 0;

                while (digitIndex < line.Count())
                {
                    string currentNumber = "";
                    char c;

                    while (Char.IsDigit(c = line[digitIndex++]))
                    {
                        currentNumber += c;
                    }

                    map[i][j] = Int32.Parse(currentNumber);
                    j++;
                }
                    
            }

            List<int> path = new List<int>();

            while (!(line = lines[i++]).Equals("PATH")) ;
            while (!(line = lines[i++]).Equals("EOF"))
            {
                path.Add(Int32.Parse(line));
            }
            this.map.RenderMap(map, path, n, m);
        }

        private void gridDropFileHandler(object sender, DragEventArgs e)
        {
            string[] filenames = (string[])e.Data.GetData(DataFormats.FileDrop, true);
            mapFromFile(filenames[0]);
        }

        /*******************************************************************/
        /************************* DRAG AND DRAW ***************************/
        /*******************************************************************/
        private void automatonImage_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            isLeftDrag = true;
        }

        private void automatonImage_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            isLeftDrag = false;
            System.Windows.Point point = e.GetPosition(e.Source as FrameworkElement);
            map.FillCellByImagePoint(point, 1);
        }

        private void automatonImage_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            isRightDrag = true;
        }

        private void automatonImage_MouseRightButtonUp(object sender, MouseButtonEventArgs e)
        {
            isRightDrag = false;
            System.Windows.Point point = e.GetPosition(e.Source as FrameworkElement);
            map.FillCellByImagePoint(point, 0);
        }

        private void automatonImage_MouseMove(object sender, MouseEventArgs e)
        {
            System.Windows.Point point = e.GetPosition(e.Source as FrameworkElement);
            // Draw alive cell
            if (isLeftDrag)
            {
                map.FillCellByImagePoint(point, 1);
            }
            // Draw dead cell
            else if (isRightDrag)
            {
                map.FillCellByImagePoint(point, 0);
            }
            // Move scrollviewer
            else if (isMiddleDrag)
            {
                double deltaX;
                double deltaY;

                deltaX = middleDragStartPoint.X - point.X;
                deltaY = middleDragStartPoint.Y - point.Y;

                gridScollViewer.ScrollToHorizontalOffset(gridScollViewer.HorizontalOffset + deltaX * middleDragFactor);
                gridScollViewer.ScrollToVerticalOffset(gridScollViewer.VerticalOffset + deltaY * middleDragFactor);
            }
        }


        /*******************************************************************/
        /************************* MIDDLE MOUSE ****************************/
        /*******************************************************************/

        private void automatonImage_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Middle)
            {
                isMiddleDrag = true;
                middleDragStartPoint = e.GetPosition(e.Source as FrameworkElement);
            }
        }

        private void automatonImage_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Middle)
            {
                isMiddleDrag = false;
            }
        }

        private void image_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            double lastZoomValue = zoomValue;
            if (e.Delta > 0)
            {
                zoomValue += zoomFactor;
            }
            else
            {
                zoomValue -= zoomFactor;
            }

            if (!map.ScaleImage(zoomValue))
                zoomValue = lastZoomValue;
        }


        /*******************************************************************/
        /**************************** BUTTONS ******************************/
        /*******************************************************************/


        /*******************************************************************/
        /**************************** COMMON *******************************/
        /*******************************************************************/

        private void Grid_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ClickCount == 2)
            {
                if (!isMaximized)
                    MaximizeWindow();
                else
                    Restore();
            }
            else
            {
                DragMove();
            }
        }

        private void MaximizeWindow()
        {
            isMaximized = true;
            _restoreLocation = new Rect { Width = Width, Height = Height, X = Left, Y = Top };

            System.Windows.Forms.Screen currentScreen;
            currentScreen = System.Windows.Forms.Screen.FromPoint(System.Windows.Forms.Cursor.Position);

            Height = currentScreen.WorkingArea.Height + 3;
            Width = currentScreen.WorkingArea.Width + 3;

            Left = currentScreen.WorkingArea.X - 2;
            Top = currentScreen.WorkingArea.Y - 2;
        }

        private void Restore()
        {
            isMaximized = false;
            Height = _restoreLocation.Height;
            Width = _restoreLocation.Width;
            Left = _restoreLocation.X;
            Top = _restoreLocation.Y;
        }

    }
}
