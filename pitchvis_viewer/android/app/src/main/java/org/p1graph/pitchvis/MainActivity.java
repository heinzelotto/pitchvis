package org.p1graph.pitchvis;

import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;

public class MainActivity extends NativeActivity {

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (!isFinishing()) {
            // App is being minimized, kill it. Winit doesn't notify us fast enough and we
            // can't handle a minimize gracefully. So we just kill the app here.
            android.os.Process.killProcess(android.os.Process.myPid());
        }
    }
}
