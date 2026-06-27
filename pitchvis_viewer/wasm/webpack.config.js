const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');
const CopyPlugin = require('copy-webpack-plugin');

// NOTE: the wasm crate is compiled (wasm-pack) and optimized (wasm-opt) by
// `cargo xtask build wasm` *before* webpack runs; webpack only bundles the prebuilt
// `./pkg`. The wasm-pack webpack plugin was removed because it ignored the custom
// `web-release` cargo profile's wasm-opt config and shipped un-optimized wasm. Always
// build via `cargo xtask build wasm[ --release]`, then `npm run serve` to preview.

module.exports = {
    entry: ['./index.js'],
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: '[name].js',
        clean: true, // purge stale hashed assets (e.g. old *.module.wasm) so dist/ isn't bloated
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
