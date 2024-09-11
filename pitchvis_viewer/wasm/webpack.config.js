const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
    entry: ['./index.js'],
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: '[name].js',
    },

    plugins: [
        new HtmlWebpackPlugin({
            template: 'index.html'
        }),
        new CopyPlugin({
            patterns: [
                { from: '../assets', to: 'static_js/assets' },
            ]
        }),
        new WasmPackPlugin({
            crateDirectory: path.resolve(__dirname, "."),
            // can't use custom profile yet https://github.com/rustwasm/wasm-pack/pull/1428
            // extraArgs: "--profile web-release",
	        // forceMode: "development", // to make sure no --profile=release is passed to wasm-pack
	        forceMode: "release",
            // extraArgs: "--no-opt", // for testing
        }),
        // Have this example work in Edge which doesn't ship `TextEncoder` or
        // `TextDecoder` at this time.
        new webpack.ProvidePlugin({
            TextDecoder: ['text-encoding', 'TextDecoder'],
            TextEncoder: ['text-encoding', 'TextEncoder']
        }),
    ],
    devServer: {
        static: {
          directory: path.join(__dirname, 'static_js'),
        },
      },
    // mode: 'development',
    mode: 'production',
    experiments: {
        asyncWebAssembly: true,
      }
};
