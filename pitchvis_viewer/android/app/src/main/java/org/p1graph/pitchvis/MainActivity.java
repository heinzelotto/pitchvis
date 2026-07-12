package org.p1graph.pitchvis;

import com.google.androidgamesdk.GameActivity;
import android.content.Intent;
import android.os.Bundle;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;
import android.os.Build.VERSION;
import android.os.Build.VERSION_CODES;
import android.view.View;
import android.view.WindowManager;

public class MainActivity extends GameActivity {

    static {
        // Load the STL first to workaround issues on old Android versions:
        // "if your app targets a version of Android earlier than Android 4.3
        // (Android API level 18),
        // and you use libc++_shared.so, you must load the shared library before any other
        // library that depends on it."
        // See https://developer.android.com/ndk/guides/cpp-support#shared_runtimes
        //System.loadLibrary("c++_shared");

        // Load the native library.
        // The name "android-game" depends on your CMake configuration, must be
        // consistent here and inside AndroidManifect.xml
        System.loadLibrary("pitchvis_viewer_lib");
    }

    private void hideSystemUI() {
        // Draw edge-to-edge: the game SurfaceView fills the ENTIRE window, including the
        // area behind the status/navigation bars and any display cutout. This is essential
        // for correct input on GameActivity: touches are delivered in surface-local
        // coordinates, so if the surface were inset by the status-bar height (the default
        // when decor "fits" system windows) every touch would land offset by that height.
        // That is exactly the split-screen bug (white bar at top, touches shifted down).
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);

        // Put the game behind any cutouts/waterfalls on devices that have them, so the
        // corresponding insets are non-zero and we render under them.
        if (VERSION.SDK_INT >= VERSION_CODES.P) {
            getWindow().getAttributes().layoutInDisplayCutoutMode
                    = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
        }
        // From API 30 onwards, this is the recommended way to hide the system UI, rather than
        // using View.setSystemUiVisibility. Hide BOTH bars (systemBars = status + navigation);
        // let them reappear transiently on a swipe from the edge.
        View decorView = getWindow().getDecorView();
        WindowInsetsControllerCompat controller = new WindowInsetsControllerCompat(getWindow(),
                decorView);
        controller.hide(WindowInsetsCompat.Type.systemBars());
        controller.setSystemBarsBehavior(
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        hideSystemUI();
    }

    @Override
    protected void onStop() {
        super.onStop();
        // if (!isFinishing()) {
        //     // App is being minimized, kill it. Winit doesn't notify us fast enough and we
        //     // can't handle a minimize gracefully. So we just kill the app here.
        //     android.os.Process.killProcess(android.os.Process.myPid());
        // }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        // The system re-shows the bars whenever we regain focus (returning from the
        // recents/permission dialog, resizing or entering/leaving split-screen, etc.),
        // so re-assert the immersive state each time.
        if (hasFocus) {
            hideSystemUI();
        }
    }
}
