"""Private copies of third-party packages used by arabic_ocr.

Everything in here is imported under the ``arabic_ocr._vendor`` namespace only.
Nothing is added to ``sys.path`` and nothing is registered under the original
package names, so these copies never interfere with packages the user has
installed. Licenses are in ``arabic_ocr/_vendor/licenses/``.
"""
