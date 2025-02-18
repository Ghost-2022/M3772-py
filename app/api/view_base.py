from flask import views, typing as ft, request, current_app, jsonify
import typing as t


class ViewBase(views.MethodView):

    def dispatch_request(self, **kwargs: t.Any) -> ft.ResponseReturnValue:
        meth = getattr(self, request.method.lower(), None)

        # If the request method is HEAD and we don't have a handler for it
        # retry with GET.
        if meth is None and request.method == "HEAD":
            meth = getattr(self, "get", None)

        assert meth is not None, f"Unimplemented method {request.method!r}"
        return self._make_response(current_app.ensure_sync(meth)(**kwargs))

    @staticmethod
    def _make_response(data):
        if isinstance(data, tuple):
            resp = {'code': data[0], 'msg': data[1]}
            if len(data) == 3:
                resp['data'] = data[2]
        else:
            resp = {'code': 0, 'msg': 'Success'}
            if data:
                resp['data'] = data
        return jsonify(resp)
