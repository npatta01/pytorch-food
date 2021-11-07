package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import androidx.appcompat.app.AppCompatActivity;
import android.widget.ListView;
import android.widget.ArrayAdapter;
import java.util.AbstractMap.SimpleEntry;

public class MainActivity extends AppCompatActivity {
//
//  private String imagePath = "image.jpg";
//  private String modelName = "model.pt";
//
//  private String[] modelClasses = ImageNetClasses.IMAGENET_CLASSES;



  private String imagePath = "food_apple_pie.jpg";
  private String modelName = "model.pt";

  private String[] modelClasses = ModelClasses.CLASSES;


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open(imagePath));
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      module = LiteModuleLoader.load(assetFilePath(this, modelName));
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

    // running the model
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    // getting tensor content as java array of floats
    final float[] scores = outputTensor.getDataAsFloatArray();

    ArrayList<Map.Entry<String, Float>> predictions = new ArrayList<>();
    for (int k = 0; k < scores.length; k++) {
      Map.Entry<String, Float> entry = new AbstractMap.SimpleEntry<String, Float>(modelClasses[k], scores[k]);
      predictions.add(entry);

    }

    Collections.sort(predictions, new Comparator<Map.Entry<String, Float>>() {
      @Override
      public int compare(
              Map.Entry<String, Float> e1,
              Map.Entry<String, Float> e2) {
        return e2.getValue().compareTo(e1.getValue());
      }
    });

    ListView listView=(ListView)findViewById(R.id.listView);
    String[] listItem = new String[]{"milk","apple"};
    listItem[0] = predictions.get(0).getKey();
    listItem[1] = predictions.get(1).getKey();



    final ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
            android.R.layout.simple_list_item_1, android.R.id.text1, listItem);
    listView.setAdapter(adapter);

  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
