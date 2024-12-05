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
                { from: '../assets', to: 'assets' },
            ]
        }),
        new WasmPackPlugin({
            crateDirectory: path.resolve(__dirname, ".."),
            extraArgs: "--profile web-release",
            // extraArgs: "--no-opt", // for testing
            outDir: "wasm/pkg",
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
