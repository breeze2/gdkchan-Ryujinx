using Ryujinx.Graphics.GAL.Multithreading.Model;
using System;

namespace Ryujinx.Graphics.GAL.Multithreading.Commands
{
    struct SetViewportsCommand : IGALCommand, IGALCommand<SetViewportsCommand>
    {
        public CommandType CommandType => CommandType.SetViewports;
        private SpanRef<Viewport> _viewports;
        private bool _disableTransform;
        private bool _yNegate;

        public void Set(SpanRef<Viewport> viewports, bool disableTransform, bool yNegate)
        {
            _viewports = viewports;
            _disableTransform = disableTransform;
            _yNegate = yNegate;
        }

        public static void Run(ref SetViewportsCommand command, ThreadedRenderer threaded, IRenderer renderer)
        {
            ReadOnlySpan<Viewport> viewports = command._viewports.Get(threaded);
            renderer.Pipeline.SetViewports(viewports, command._disableTransform, command._yNegate);
            command._viewports.Dispose(threaded);
        }
    }
}
