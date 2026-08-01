(this.webpackJsonpstreamlit_component_template =
  this.webpackJsonpstreamlit_component_template || []).push([
  [0],
  {
    15: function (module, exports, require) {
      module.exports = require(30);
    },
    30: function (module, exports, require) {
      "use strict";
      require.r(exports);

      var React = require(5);
      var ReactDefault = require.n(React);
      var ReactDOM = require(12);
      var ReactDOMDefault = require.n(ReactDOM);
      var classCallCheck = require(0);
      var inherits = require(1);
      var createSuper = require(2);
      var Streamlit = require(8);
      var Plot = require(14);
      var PlotDefault = require.n(Plot);

      var Viewer = (function (Base) {
        inherits.a(Component, Base);
        var superConstructor = createSuper.a(Component);

        function Component() {
          var self;
          classCallCheck.a(this, Component);
          self = superConstructor.apply(this, arguments);
          self.interactionMode = "pan";
          self.eventCounter = 0;

          self.emitMode = function (mode) {
            self.interactionMode = mode;
            Streamlit.a.setComponentValue(
              JSON.stringify({
                type: "mode",
                mode: mode,
                eventId: Date.now() + "-" + self.eventCounter++,
              })
            );
          };

          self.handleClick = function (event) {
            if (self.interactionMode !== "select" || !event || !event.points) {
              return;
            }
            var points = event.points.map(function (point) {
              return {
                x: point.x,
                y: point.y,
                curveNumber: point.curveNumber,
                pointNumber: point.pointNumber,
                pointIndex: point.pointIndex,
                customdata: point.customdata,
              };
            });
            Streamlit.a.setComponentValue(
              JSON.stringify({
                type: "click",
                mode: self.interactionMode,
                eventId: Date.now() + "-" + self.eventCounter++,
                points: points,
              })
            );
          };

          self.bindModebar = function (figure, plot) {
            if (!plot || !plot.querySelector) return;
            var selectButton = plot.querySelector('[data-title="Select"]');
            var panButton = plot.querySelector('[data-title="Pan"]');
            var zoomButton = plot.querySelector('[data-title="Zoom"]');
            if (selectButton) {
              selectButton.style.background =
                self.interactionMode === "select" ? "rgba(80, 150, 255, 0.32)" : "";
              selectButton.style.borderRadius = "3px";
              if (!selectButton.dataset.spalignBound) {
                selectButton.dataset.spalignBound = "true";
                selectButton.addEventListener(
                  "click",
                  function () {
                    self.emitMode("select");
                  },
                  true
                );
              }
            }
            if (panButton) {
              panButton.style.background =
                self.interactionMode === "pan" ? "rgba(80, 150, 255, 0.32)" : "";
              panButton.style.borderRadius = "3px";
              if (!panButton.dataset.spalignBound) {
                panButton.dataset.spalignBound = "true";
                panButton.addEventListener(
                  "click",
                  function () {
                    self.emitMode("pan");
                  },
                  true
                );
              }
            }
            if (zoomButton) {
              zoomButton.style.background =
                self.interactionMode === "zoom" ? "rgba(80, 150, 255, 0.32)" : "";
              zoomButton.style.borderRadius = "3px";
              if (!zoomButton.dataset.spalignBound) {
                zoomButton.dataset.spalignBound = "true";
                zoomButton.addEventListener(
                  "click",
                  function () {
                    self.emitMode("zoom");
                  },
                  true
                );
              }
            }
          };

          self.render = function () {
            var figure = JSON.parse(self.props.args.plot_obj);
            var height = self.props.args.override_height;
            var width = self.props.args.override_width;
            self.interactionMode = self.props.args.current_mode || "pan";
            var config = Object.assign({}, figure.config || {});
            var selectButton = {
              name: "Select",
              icon: {
                width: 24,
                height: 24,
                path: "M3 2L3 18L8 13L12 22L16 20L12 12L19 12Z",
              },
              click: function () {
                self.emitMode("select");
              },
            };

            config.modeBarButtons = [[
              "zoom2d",
              "pan2d",
              selectButton,
              "resetScale2d",
              "toImage",
            ]];
            figure.layout.dragmode =
              self.interactionMode === "select"
                ? false
                : self.interactionMode === "zoom"
                ? "zoom"
                : "pan";

            Streamlit.a.setFrameHeight(height);
            return ReactDefault.a.createElement(PlotDefault.a, {
              data: figure.data,
              layout: figure.layout,
              config: config,
              frames: figure.frames,
              onClick: self.handleClick,
              onInitialized: self.bindModebar,
              onUpdate: self.bindModebar,
              style: { width: width, height: height },
              className: "stPlotlyChart",
            });
          };

          return self;
        }

        return Component;
      })(Streamlit.b);

      var ConnectedViewer = Streamlit.c(Viewer);
      ReactDOMDefault.a.render(
        ReactDefault.a.createElement(
          ReactDefault.a.StrictMode,
          null,
          ReactDefault.a.createElement(ConnectedViewer, null)
        ),
        document.getElementById("root")
      );
    },
  },
  [[15, 1, 2]],
]);
