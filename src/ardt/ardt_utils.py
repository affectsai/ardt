#  Copyright (c) 2025. Affects AI LLC
#
#  Licensed under the Creative Common CC BY-NC-SA 4.0 International License (the "License");
#  you may not use this file except in compliance with the License. The full text of the License is
#  provided in the included LICENSE file. If this file is not available, you may obtain a copy of the
#  License at
#
#       https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing permissions and limitations
#  under the License.

import functools
import warnings

def ardt_deprecated(msg):
    def actual_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper docstring"""
            warnings.warn(f"Calling deprecated function {func.__name__}. {msg}",
                          category=DeprecationWarning)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return actual_decorator