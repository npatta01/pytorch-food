package org.pytorch.helloworld;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.text.SpannableStringBuilder;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;

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
import java.time.Instant;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.text.HtmlCompat;

import android.widget.ListView;
import android.widget.ArrayAdapter;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
//
//  private String imagePath = "image.jpg";
//  private String modelName = "model.pt";
//
//  private String[] modelClasses = ImageNetClasses.IMAGENET_CLASSES;



  private String imagePath = "food_apple_pie.jpg";
  private String modelName = "model.pt";
  private Module module = null;
  private String[] modelClasses = ModelClasses.CLASSES;


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    ImageButton button = (ImageButton) findViewById(R.id.btn_picture);
    button.setOnClickListener(new View.OnClickListener() {
      public void onClick(View v) {
        processAction();
      }
    });

    try {
      module = LiteModuleLoader.load(assetFilePath(this, modelName));
    } catch (IOException e) {
      e.printStackTrace();
    }

    //processAction(null);

  }

  private void processAction() {
    Bitmap bitmap = null;

    long imageLoadStart = System.currentTimeMillis();
    long imageLoadEnd =0;

    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = getBitmap();
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      // module = LiteModuleLoader.load(assetFilePath(this, modelName));

      imageLoadEnd = System.currentTimeMillis();
    } catch (IOException e) {
      Log.e("PytorchHelloWorld", "Error reading assets", e);
      //finish();
    }


    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    long preprocessStart = System.currentTimeMillis();
    long preprocessEnd  ;

    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
    preprocessEnd = System.currentTimeMillis();

    // running the model
    long inferenceStart = System.currentTimeMillis();
    long inferenceEnd ;
    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
    inferenceEnd = System.currentTimeMillis();

    // getting tensor content as java array of floats
    long postProcessStart = System.currentTimeMillis();

    final float[] scores = outputTensor.getDataAsFloatArray();

    final float[] probabilities = softmax(scores);
    long postProcessEnd = System.currentTimeMillis();


    ArrayList<Map.Entry<String, Float>> predictions = new ArrayList<>();
    for (int k = 0; k < probabilities.length; k++) {
      Map.Entry<String, Float> entry = new AbstractMap.SimpleEntry<String, Float>(modelClasses[k], probabilities[k]);
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

    ListView listView=(ListView)findViewById(R.id.predictionListView);
    String[] listItem = new String[]{"milk","apple","shoes"};

    for (int i=0; i <= 2; i++){
      @SuppressLint("DefaultLocale") String label = String.format("%s  (%.2f%%)",predictions.get(i).getKey(),predictions.get(i).getValue()*100 );
      listItem[i]  = label;
    }



    final ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
            android.R.layout.simple_list_item_1, android.R.id.text1, listItem);
    listView.setAdapter(adapter);

    SpannableStringBuilder formattedText = new SpannableStringBuilder();

    TextView timingContentView=(TextView)findViewById(R.id.timingContent);

    formattedText.append(String.format("Load Image: %s ms", imageLoadEnd-imageLoadStart)).append("\n");
    formattedText.append(String.format("Preprocess: %s ms", preprocessEnd-preprocessStart)).append("\n");
    formattedText.append(String.format("Inference: %s ms", inferenceEnd-inferenceStart)).append("\n");
    formattedText.append(String.format("PostProcess: %s ms", postProcessEnd-postProcessStart)).append("\n");


//    String formattedText = "<b>Load Image</b> : 5ms <br>" +
//            "<b>Load Image</b> : 5ms <br>" +
//            "<b>Preprocess</b> : 5ms <br>" +
//            "<b>Inference</b> : 5ms <br>" +
//            "<b>PostProcess</b> : 5ms <br>" ;

//    timingContentView.setText(HtmlCompat.fromHtml(formattedText, HtmlCompat.FROM_HTML_MODE_LEGACY));
    timingContentView.setText(formattedText);


  }

  private Bitmap getBitmap() throws IOException {
    return BitmapFactory.decodeStream(getAssets().open(imagePath));
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
  private float[] softmax(float[] neuronValues) {
    float total = 0f;
    for (float f: neuronValues){
      total+=Math.exp(f);
    }

    float[] result = neuronValues.clone();

    for (int i=0 ; i < neuronValues.length; i++){
      result[i] = (float) (Math.exp(neuronValues[i]) / total);
    }

    return result;
  }
}
