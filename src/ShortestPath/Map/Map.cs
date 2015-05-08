using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace MapGUI
{
    /// <summary>
    ///     Draw the automaton
    /// </summary>
    public class Map
    {
        /******************************************************************/
        /******************* PROPERTIES, PRIVATE FIELDS *******************/
        /******************************************************************/
        /// <summary>
        ///     The DPI constants
        /// </summary>
        private const double DPI_X = 300.0;
        private const double DPI_Y = 300.0;

        /// <summary>
        ///     Color of the nest
        /// </summary>
        private System.Drawing.Color NEST_COLOR = System.Drawing.Color.FromArgb(200, 20, 20);

        private System.Drawing.Color PATH_COLOR = System.Drawing.Color.FromArgb(0, 255, 0);

        /// <summary>
        ///     Dimensions of a cell in pixels
        /// </summary>
        private const int CELL_WIDTH = 10;
        private const int CELL_HEIGHT = 10;

        private int widthPixels;
        private int heightPixels;

        private int stride;
        private int bytesPerPixel;

        /// <summary>
        ///     Actual bitmap of the automaton
        /// </summary>
        private WriteableBitmap wBitmap;

        public WriteableBitmap Bitmap
        {
            get { return wBitmap; }
            private set { wBitmap = value; }
        }

        /// <summary>
        ///     Map and its dimensions
        /// </summary>
        int[][] map;
        int n;
        int m;

        List<int> shortestPath;

        /// <summary>
        ///     The destination of our bitmap
        /// </summary>
        private Image imageDest;


        double originalImageWidth;
        double originalImageHeight;

        const double MAX_ZOOM_SCALE = 10.0;
        const double MIN_ZOOM_SCALE = 4.0;

        /******************************************************************/
        /************************** CONSTRUCTORS **************************/
        /******************************************************************/

        /// <summary>
        ///     
        /// </summary>
        /// <param name="width">
        ///     Number of automaton cells in a row, NOT in pixels
        /// </param>
        /// <param name="height">
        ///     
        /// </param>
        /// <param name="automaton"></param>
        /// <param name="imageDestionation"></param>
        public Map(Image imageDestionation)
        {
            this.imageDest = imageDestionation;

            //init();
        }

        /*******************************************************************/
        /************************ PRIVATE METHODS **************************/
        /*******************************************************************/

        private void init()
        {
            initBitmap();
            renderBitmap();
            drawPath();
        }

        private unsafe void initBitmap()
        {
            int bitmapWidth = n* CELL_WIDTH + n + 1;
            int bitmapHeight = m * CELL_HEIGHT + m + 1;

            wBitmap = new WriteableBitmap(bitmapWidth, bitmapHeight, DPI_X, DPI_Y, PixelFormats.Rgb24, null);

            imageDest.Width = bitmapWidth;
            imageDest.Height = bitmapHeight;

            originalImageWidth = imageDest.Width;
            originalImageHeight = imageDest.Height;

            imageDest.Source = wBitmap;

            widthPixels = wBitmap.PixelWidth;
            heightPixels = wBitmap.PixelHeight;
            stride = wBitmap.BackBufferStride;
            bytesPerPixel = (wBitmap.Format.BitsPerPixel) / 8;
        }

        private unsafe void renderBitmap()
        {
            wBitmap.Lock();
            byte* pImgData = (byte*)wBitmap.BackBuffer;

            int cRowStart = 0;
            int cColStart = 0;
            for (int row = 0; row < heightPixels; row++)
            {
                cColStart = cRowStart;
                for (int col = 0; col < widthPixels; col++)
                {
                    byte* bPixel = pImgData + cColStart;

                    // draw the nest
                    if ((col % (CELL_WIDTH + 1) == 0) || (row % (CELL_HEIGHT + 1) == 0) || col == widthPixels - 1 || row == heightPixels - 1)
                    {
                        //int cellI = (row ) / (CELL_WIDTH);
                        //int cellJ = (col ) / (CELL_HEIGHT);

                        int cellI = (row - 1) / (CELL_WIDTH + 1);
                        int cellJ = (col - 1) / (CELL_HEIGHT + 1);

                        if (belongsToPath(cellI, cellJ))
                        {
                            bPixel[0] = PATH_COLOR.R;
                            bPixel[1] = PATH_COLOR.G;
                            bPixel[2] = PATH_COLOR.B;
                        }
                        else
                        {
                            bPixel[0] = NEST_COLOR.R;
                            bPixel[1] = NEST_COLOR.G;
                            bPixel[2] = NEST_COLOR.B;
                        }
                    }
                    // draw the cell
                    else
                    {
                        // get cell indecies
                        int cellI = (row - 1) / (CELL_WIDTH + 1);
                        int cellJ = (col - 1) / (CELL_HEIGHT + 1);

                        int intR = (map[cellI][cellJ] *10);
                        if(intR > 255)
                            intR = 255;
                        byte R = (byte)intR;

                        int intG = (map[cellI][cellJ] * 8);
                        if (intG > 255)
                            intG = 255;
                        byte G = (byte)intG;

                        int intB = (map[cellI][cellJ] * 4);
                        if (intB > 255)
                            intB = 255;
                        byte B = (byte)intB;

                        bPixel[0] = R;//(byte)((map[cellI][cellJ] * 5) % 255 );
                        bPixel[1] = 0;// (byte)((map[cellI][cellJ]) % 255);
                        bPixel[2] = 0;//(byte)((map[cellI][cellJ] * 5) % 255);
                    }
                    cColStart += bytesPerPixel;
                }
                cRowStart += stride;
            }
            Int32Rect rect = new Int32Rect(0, 0, widthPixels, heightPixels);
            wBitmap.AddDirtyRect(rect);
            wBitmap.Unlock();
        }

        private void drawPath()
        {

        }

        private bool belongsToPath(int i, int j)
        {
            int index = i * n + j;
            foreach (int v in shortestPath)
            {
                if (v == index)
                    return true;
            }

            return false;
        }

        /*******************************************************************/
        /************************* PUBLIC METHODS **************************/
        /*******************************************************************/

        public void RenderMap(int[][] map, List<int> shortestPath, int n, int m)
        {
            this.map = map;
            this.n = n;
            this.m = m;

            this.shortestPath = shortestPath;

            init();
        }

        /// <summary>
        ///     Fills a cell with given color
        /// </summary>
        /// <param name="cellRow"></param>
        /// <param name="cellCol"></param>
        /// <param name="c"></param>
        public unsafe void FillCell(int cellRow, int cellCol, int state)
        {
            int startPixelI = ((cellRow * CELL_WIDTH) + cellRow + 1);
            int startPixelJ = ((cellCol * CELL_HEIGHT) + cellCol + 1);

            // todo global states
            int width = wBitmap.PixelWidth;
            int height = wBitmap.PixelHeight;
            int stride = wBitmap.BackBufferStride;
            int bytesPerPixel = (wBitmap.Format.BitsPerPixel) / 8;

            wBitmap.Lock();
            byte* pImgData = (byte*)wBitmap.BackBuffer;

            Int32Rect rect = new Int32Rect(startPixelJ, startPixelI, CELL_WIDTH, CELL_HEIGHT);

            byte* startPixel = pImgData +
                                (stride * (startPixelI + 1)) +
                                ((startPixelJ + 1) * (bytesPerPixel));

            for (int x = 0; x < CELL_WIDTH; x++)
            {
                for (int y = 0; y < CELL_HEIGHT; y++)
                {
                    startPixel = pImgData +
                                (stride * (startPixelI + x)) +
                                (bytesPerPixel * (startPixelJ + y));
                    try
                    {
                        // color the bitmap
                        /*
                        System.Drawing.Color c = GridSettings.GetStateColor(state);
                        startPixel[0] = c.R;
                        startPixel[1] = c.G;
                        startPixel[2] = c.B;*/
                    }
                    catch (AccessViolationException e) { Console.Write(e.StackTrace); }
                }
            }
            try
            {
                wBitmap.AddDirtyRect(rect);
            }
            catch (ArgumentException e) { Console.Write(e.StackTrace); }
            wBitmap.Unlock();
        }

        /// <summary>
        ///     Takes point from the image
        ///     Scales it to proper index for bitmap
        ///     and fills a proper cell with given color
        /// </summary>
        /// <param name="point"></param>
        /// <param name="c"></param>
        public void FillCellByImagePoint(System.Windows.Point point, int state)
        {
            int width = wBitmap.PixelWidth;
            int height = wBitmap.PixelHeight;

            double actualWidth = imageDest.ActualWidth;
            double actualHeight = imageDest.ActualHeight;

            double scaleWidth = actualWidth / width;
            double scaleHeight = actualHeight / height;

            // scalled pixel indecies image
            int pixelI = (int)(point.X / scaleWidth);
            int pixelJ = (int)(point.Y / scaleHeight);

            // cell indecies
            int cellI = (pixelI - 1) / (CELL_WIDTH + 1);
            int cellJ = (pixelJ - 1) / (CELL_HEIGHT + 1);

            //automaton.Grid.SetState(cellJ, cellI, state);
        }

        /// <summary>
        ///     Resets the bitmap.
        ///     Has to be called after resizing automaton.
        ///     Currectly, does not save previous state.
        /// </summary>
        public void Reset()
        {
            init();
        }

        public bool ScaleImage(double scale)
        {
            try
            {
                double width = originalImageWidth * scale;
                double height = originalImageHeight * scale;


                if ((width < originalImageWidth / MIN_ZOOM_SCALE && height < originalImageHeight / MIN_ZOOM_SCALE) ||
                    (width > originalImageWidth * MAX_ZOOM_SCALE && height > originalImageHeight * MAX_ZOOM_SCALE))
                    return false;

                imageDest.Width = width;
                imageDest.Height = height;

            }
            catch (Exception e) { Console.Write(e.StackTrace); }

            return true;
        }
    }
}
